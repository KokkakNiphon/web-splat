#[cfg(feature = "npz")]
use std::io::BufReader;
use std::io::{Read, Seek};
use std::path::PathBuf;

use bytemuck::Zeroable;
use cgmath::{Array, EuclideanSpace, InnerSpace, Point3, Vector3};
use half::f16;

use crate::pointcloud::{Aabb, Covariance3D, Gaussian, GaussianCompressed, GaussianQuantization};

#[cfg(feature = "npz")]
use self::npz::NpzReader;

use self::ply::PlyReader;
use self::sog::SogReader;

#[cfg(feature = "npz")]
pub mod npz;
pub mod ply;
pub mod sog;

pub trait PointCloudReader {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error>;

    fn magic_bytes() -> &'static [u8];
    fn file_ending() -> &'static str;
}

pub enum InputSource<R: Read + Seek> {
    File(R),
    Path(PathBuf),
}

pub struct GenericGaussianPointCloud {
    gaussians: Vec<u8>,
    sh_coefs: Vec<u8>,
    compressed: bool,
    pub covars: Option<Vec<Covariance3D>>,
    pub quantization: Option<GaussianQuantization>,
    pub sh_deg: u32,
    pub num_points: usize,
    pub kernel_size: Option<f32>,
    pub mip_splatting: Option<bool>,
    pub background_color: Option<[f32; 3]>,

    pub up: Option<Vector3<f32>>,
    pub center: Point3<f32>,
    pub aabb: Aabb<f32>,
}

impl GenericGaussianPointCloud {
    pub fn load<R: Read + Seek + Send + Sync>(
        input: InputSource<R>,
    ) -> Result<Self, anyhow::Error> {
        match input {
            InputSource::File(mut f) => {
                let mut signature: [u8; 4] = [0; 4];
                f.read_exact(&mut signature)?;
                f.rewind()?;
                if signature.starts_with(PlyReader::<R>::magic_bytes()) {
                    let mut ply_reader = PlyReader::new(f)?;
                    return ply_reader.read();
                }
                // Both SOG and NPZ are zip files.
                // We check if it is a SOG file by looking for "meta.json"
                if signature.starts_with(SogReader::magic_bytes()) {
                    let is_sog = {
                        let mut archive = zip::ZipArchive::new(&mut f);
                        match archive {
                            Ok(mut a) => a.by_name("meta.json").is_ok(),
                            Err(_) => false,
                        }
                    };
                    f.rewind()?;

                    if is_sog {
                        let mut sog_reader = SogReader::new_zip(f)?;
                        return sog_reader.read();
                    }

                    #[cfg(feature = "npz")]
                    {
                        let mut reader = BufReader::new(f);
                        let mut npz_reader = NpzReader::new(&mut reader)?;
                        return npz_reader.read();
                    }
                }
                return Err(anyhow::anyhow!("Unknown file format"));
            }
            InputSource::Path(path) => {
                if path.is_dir() {
                    let mut sog_reader = SogReader::new(path)?;
                    return sog_reader.read();
                } else {
                    // Fallback to opening file
                    let f = std::fs::File::open(path)?;
                    // But R needs to be std::fs::File?
                    // The generic R is defined by the caller.
                    // If InputSource::Path is used, R is not really used in this branch.
                    // However, to convert to InputSource::File(f), f must match R.
                    // This only works if R = std::fs::File.
                    // If R is something else, we cannot open a path into R.

                    // Helper: recursion requires R to be File.
                    // Since we can't ensure R is File, we can't recursively call load with File(f).
                    // We should implement load for Path separately or require R to be inferred?

                    // Solution: Just call load(InputSource::File(std::fs::File::open(path)?)) BUT
                    // This requires R to be File.
                    // If the caller calls load::<File>(Path(...)), it works.
                    // If the caller calls load::<Cursor>(Path(...)), it fails to compile/run this branch?
                    // Actually, we can just error if it's not a directory because SOG must be a directory.
                    // If it's a file, the caller should have opened it or we update usage.

                    return Err(anyhow::anyhow!("Only directory paths (SOG) are supported via InputSource::Path. For files, open them and use InputSource::File."));
                }
            }
        }
    }

    fn new(
        gaussians: Vec<Gaussian>,
        sh_coefs: Vec<[[f16; 3]; 16]>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::zeroed();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            gaussians
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up: up,
            center,
            aabb: bbox,
            compressed: false,
        }
    }

    #[cfg(feature = "npz")]
    fn new_compressed(
        gaussians: Vec<GaussianCompressed>,
        sh_coefs: Vec<u8>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::unit();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            gaussians
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up: up,
            center,
            aabb: bbox,
            compressed: true,
        }
    }

    pub fn gaussians(&self) -> anyhow::Result<&[Gaussian]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn gaussians_compressed(&self) -> anyhow::Result<&[GaussianCompressed]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn sh_coefs_buffer(&self) -> &[u8] {
        &self.sh_coefs
    }

    pub fn gaussian_buffer(&self) -> &[u8] {
        &self.gaussians
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }
}

// Fit a plane to a collection of points.
// Fast, and accurate to within a few degrees.
// Returns None if the points do not span a plane.
// see http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
fn plane_from_points(points: &[Point3<f32>]) -> (Point3<f32>, Option<Vector3<f32>>) {
    let n = points.len();

    let mut sum = Point3 {
        x: 0.0f32,
        y: 0.0f32,
        z: 0.0f32,
    };
    for p in points {
        sum = &sum + p.to_vec();
    }
    let centroid = &sum * (1.0 / (n as f32));
    if n < 3 {
        return (centroid, None);
    }

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for p in points {
        let r = p - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= n as f32;
    xy /= n as f32;
    xz /= n as f32;
    yy /= n as f32;
    yz /= n as f32;
    zz /= n as f32;

    let mut weighted_dir = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    {
        let det_x = yy * zz - yz * yz;
        let axis_dir = Vector3 {
            x: det_x,
            y: xz * yz - xy * zz,
            z: xy * yz - xz * yy,
        };
        let mut weight = det_x * det_x;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    {
        let det_y = xx * zz - xz * xz;
        let axis_dir = Vector3 {
            x: xz * yz - xy * zz,
            y: det_y,
            z: xy * xz - yz * xx,
        };
        let mut weight = det_y * det_y;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    {
        let det_z = xx * yy - xy * xy;
        let axis_dir = Vector3 {
            x: xy * yz - xz * yy,
            y: xy * xz - yz * xx,
            z: det_z,
        };
        let mut weight = det_z * det_z;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    let mut normal = weighted_dir.normalize();

    if normal.dot(Vector3::unit_y()) < 0. {
        normal = -normal;
    }
    if normal.is_finite() {
        (centroid, Some(normal))
    } else {
        (centroid, None)
    }
}

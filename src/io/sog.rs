use std::io::{Read, Seek};
use std::path::PathBuf;

use cgmath::{Point3, Quaternion, Vector3};
use half::f16;
use image::DynamicImage;
use serde::Deserialize;
use zip::ZipArchive;

use crate::io::{GenericGaussianPointCloud, PointCloudReader};
use crate::pointcloud::Gaussian;
use crate::utils::build_cov;

#[derive(Debug, Deserialize)]
struct MetaMinMax {
    mins: [f32; 3],
    maxs: [f32; 3],
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MetaMeans {
    mins: [f32; 3],
    maxs: [f32; 3],
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MetaScales {
    codebook: Vec<f32>,
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MetaQuats {
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MetaSh0 {
    codebook: Vec<f32>,
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MetaShN {
    count: usize,
    bands: usize,
    codebook: Vec<f32>,
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Meta {
    version: u32,
    count: usize,
    #[serde(default)]
    antialias: bool,
    means: MetaMeans,
    scales: MetaScales,
    quats: MetaQuats,
    sh0: MetaSh0,
    #[serde(rename = "shN")]
    sh_n: Option<MetaShN>,
}

pub trait ReadSeek: Read + Seek + Send + Sync {}
impl<T: Read + Seek + Send + Sync> ReadSeek for T {}

enum SogSource<'a> {
    Dir(PathBuf),
    Zip(ZipArchive<Box<dyn ReadSeek + 'a>>),
}

impl<'a> SogSource<'a> {
    fn read_file_content(&mut self, filename: &str) -> Result<Vec<u8>, anyhow::Error> {
        match self {
            SogSource::Dir(path) => {
                let p = path.join(filename);
                Ok(std::fs::read(p)?)
            }
            SogSource::Zip(archive) => {
                let mut file = archive.by_name(filename)?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;
                Ok(buffer)
            }
        }
    }

    fn read_image(&mut self, filename: &str) -> Result<DynamicImage, anyhow::Error> {
        let content = self.read_file_content(filename)?;
        Ok(image::load_from_memory(&content)?)
    }
}

pub struct SogReader<'a> {
    source: SogSource<'a>,
    meta: Option<Meta>,
}

impl<'a> SogReader<'a> {
    pub fn new(path: PathBuf) -> Result<Self, anyhow::Error> {
        Ok(Self {
            source: SogSource::Dir(path),
            meta: None,
        })
    }

    pub fn new_zip<R: Read + Seek + Send + Sync + 'a>(reader: R) -> Result<Self, anyhow::Error> {
        let boxed_reader: Box<dyn ReadSeek + 'a> = Box::new(reader);
        Ok(Self {
            source: SogSource::Zip(ZipArchive::new(boxed_reader)?),
            meta: None,
        })
    }

    fn read_meta(&mut self) -> Result<(), anyhow::Error> {
        let content = self.source.read_file_content("meta.json")?;
        self.meta = Some(serde_json::from_slice(&content)?);
        Ok(())
    }

    fn unlog(n: f32) -> f32 {
        n.signum() * (n.abs().exp() - 1.0)
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    pub fn magic_bytes() -> &'static [u8] {
        &[0x50, 0x4B, 0x03, 0x04]
    }
}

impl<'a> PointCloudReader for SogReader<'a> {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error> {
        if self.meta.is_none() {
            self.read_meta()?;
        }
        let meta = self.meta.as_ref().unwrap();

        // Load means
        let means_l_img = self.source.read_image(&meta.means.files[0])?.to_rgba8();
        let means_u_img = self.source.read_image(&meta.means.files[1])?.to_rgba8();

        let width = means_l_img.width();

        // Load scales
        let scales_img = self.source.read_image(&meta.scales.files[0])?.to_rgb8();

        // Load quats
        let quats_img = self.source.read_image(&meta.quats.files[0])?.to_rgba8();

        // Load sh0
        let sh0_img = self.source.read_image(&meta.sh0.files[0])?.to_rgba8();

        // Load shN if present
        let (shn_centroids_img, shn_labels_img, shn_codebook, shn_bands) =
            if let Some(ref shn) = meta.sh_n {
                (
                    Some(self.source.read_image(&shn.files[0])?.to_rgb8()),
                    Some(self.source.read_image(&shn.files[1])?.to_rgba8()),
                    Some(&shn.codebook),
                    shn.bands,
                )
            } else {
                (None, None, None, 0)
            };

        let num_points = meta.count;
        let mut gaussians = Vec::with_capacity(num_points);
        let mut sh_coefs = Vec::with_capacity(num_points);

        let sh_deg = if meta.sh_n.is_some() {
            meta.sh_n.as_ref().unwrap().bands as u32
        } else {
            0
        };

        for i in 0..num_points {
            let x_coord = (i as u32) % width;
            let y_coord = (i as u32) / width;

            // --- Positions ---
            let ml = means_l_img.get_pixel(x_coord, y_coord);
            let mu = means_u_img.get_pixel(x_coord, y_coord);

            let qx = ((mu[0] as u16) << 8) | ml[0] as u16;
            let qy = ((mu[1] as u16) << 8) | ml[1] as u16;
            let qz = ((mu[2] as u16) << 8) | ml[2] as u16;

            let nx = Self::lerp(meta.means.mins[0], meta.means.maxs[0], qx as f32 / 65535.0);
            let ny = Self::lerp(meta.means.mins[1], meta.means.maxs[1], qy as f32 / 65535.0);
            let nz = Self::lerp(meta.means.mins[2], meta.means.maxs[2], qz as f32 / 65535.0);

            let pos = Point3::new(Self::unlog(nx), Self::unlog(ny), Self::unlog(nz));

            // --- Scales ---
            let s_pixel = scales_img.get_pixel(x_coord, y_coord);
            let sx = meta.scales.codebook[s_pixel[0] as usize];
            let sy = meta.scales.codebook[s_pixel[1] as usize];
            let sz = meta.scales.codebook[s_pixel[2] as usize];
            let scale = Vector3::new(sx, sy, sz);

            // --- Quats ---
            let q_pixel = quats_img.get_pixel(x_coord, y_coord);

            let to_comp = |c: u8| (c as f32 / 255.0 - 0.5) * 2.0 / std::f32::consts::SQRT_2;
            let qa = to_comp(q_pixel[0]);
            let qb = to_comp(q_pixel[1]);
            let qc = to_comp(q_pixel[2]);
            let mode = q_pixel[3] - 252;

            let t = qa * qa + qb * qb + qc * qc;
            let d = (1.0 - t).max(0.0).sqrt();

            let rot = match mode {
                0 => Quaternion::new(d, qa, qb, qc),
                1 => Quaternion::new(qa, d, qb, qc),
                2 => Quaternion::new(qa, qb, d, qc),
                3 => Quaternion::new(qa, qb, qc, d),
                _ => Quaternion::new(1.0, 0.0, 0.0, 0.0), // Error fallback
            };

            // --- Opacity ---
            let sh0_pixel = sh0_img.get_pixel(x_coord, y_coord);
            let opacity = sh0_pixel[3] as f32 / 255.0;

            let cov = build_cov(rot, scale);

            gaussians.push(Gaussian::new(
                pos,
                f16::from_f32(opacity),
                cov.map(|x| f16::from_f32(x)),
            ));

            // --- SH ---
            let mut sh = [[f16::ZERO; 3]; 16];

            // DC
            sh[0][0] = f16::from_f32(meta.sh0.codebook[sh0_pixel[0] as usize]);
            sh[0][1] = f16::from_f32(meta.sh0.codebook[sh0_pixel[1] as usize]);
            sh[0][2] = f16::from_f32(meta.sh0.codebook[sh0_pixel[2] as usize]);

            // AC
            if let (Some(cent_img), Some(lbl_img), Some(codebook)) = (
                shn_centroids_img.as_ref(),
                shn_labels_img.as_ref(),
                shn_codebook,
            ) {
                let lbl_pixel = lbl_img.get_pixel(x_coord, y_coord);
                let idx = lbl_pixel[0] as usize + ((lbl_pixel[1] as usize) << 8);

                let stride = match shn_bands {
                    1 => 3,
                    2 => 8,
                    3 => 15,
                    _ => 0,
                };

                for k in 0..stride {
                    let u = (idx as u32 % 64) * stride as u32 + k as u32;
                    let v = idx as u32 / 64;

                    let p = cent_img.get_pixel(u, v);

                    sh[k + 1][0] = f16::from_f32(codebook[p[0] as usize]);
                    sh[k + 1][1] = f16::from_f32(codebook[p[1] as usize]);
                    sh[k + 1][2] = f16::from_f32(codebook[p[2] as usize]);
                }
            }

            sh_coefs.push(sh);
        }

        Ok(GenericGaussianPointCloud::new(
            gaussians, sh_coefs, sh_deg, num_points, None, None, None, None, None,
        ))
    }

    fn magic_bytes() -> &'static [u8] {
        &[0x50, 0x4B, 0x03, 0x04]
    }

    fn file_ending() -> &'static str {
        "sog"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlog() {
        assert_eq!(SogReader::unlog(0.0), 0.0);
        let val = 2.0;
        let expected = (2.0f32.exp() - 1.0);
        assert!((SogReader::unlog(val) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(SogReader::lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(SogReader::lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(SogReader::lerp(0.0, 10.0, 0.5), 5.0);
    }
}

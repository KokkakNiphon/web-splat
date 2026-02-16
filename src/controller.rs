use cgmath::*;
use num_traits::Float;
use std::f32::consts::PI;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use winit::keyboard::KeyCode;

use crate::camera::PerspectiveCamera;

#[derive(Debug, Clone)]
pub struct TouchState {
    pub touches: Vec<Touch>,
    pub last_touch_count: usize,
    pub last_pinch_distance: Option<f32>,
    pub last_touch_center: Option<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct Touch {
    pub id: u64,
    pub position: (f32, f32),
    pub phase: TouchPhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

impl TouchState {
    pub fn new() -> Self {
        Self {
            touches: Vec::new(),
            last_touch_count: 0,
            last_pinch_distance: None,
            last_touch_center: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlMode {
    Orbit,
    Fly,
}

#[derive(Debug)]
pub struct CameraController {
    pub center: Point3<f32>,
    pub up: Option<Vector3<f32>>,
    amount: Vector3<f32>,
    shift: Vector2<f32>,
    rotation: Vector3<f32>,
    scroll: f32,
    pub speed: f32,
    pub sensitivity: f32,

    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
    pub alt_pressed: bool,
    pub user_inptut: bool,

    // Touch support
    pub touch_state: TouchState,
    pub mode: ControlMode,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            center: Point3::origin(),
            amount: Vector3::zero(),
            shift: Vector2::zero(),
            rotation: Vector3::zero(),
            up: None,
            scroll: 0.0,
            speed,
            sensitivity,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
            alt_pressed: false,
            user_inptut: false,
            touch_state: TouchState::new(),
            mode: ControlMode::Orbit,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        let processed = match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount.z = amount;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount.z = -amount;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount.x = -amount;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount.x = amount;
                true
            }
            KeyCode::KeyQ => {
                self.rotation.z = amount / self.sensitivity;
                true
            }
            KeyCode::KeyE => {
                self.rotation.z = -amount / self.sensitivity;
                true
            }
            KeyCode::Space => {
                self.amount.y = amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount.y = -amount;
                true
            }
            _ => false,
        };
        self.user_inptut = processed;
        return processed;
    }

    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        if self.left_mouse_pressed {
            self.rotation.x += mouse_dx as f32;
            self.rotation.y += mouse_dy as f32;
            self.user_inptut = true;
        }
        if self.right_mouse_pressed {
            self.shift.y += -mouse_dx as f32;
            self.shift.x += mouse_dy as f32;
            self.user_inptut = true;
        }
    }

    pub fn process_scroll(&mut self, dy: f32) {
        self.scroll += -dy;
        self.user_inptut = true;
    }

    pub fn process_touch(&mut self, touch: Touch) {
        // Update touch state
        match touch.phase {
            TouchPhase::Started => {
                self.touch_state.touches.push(touch);
            }
            TouchPhase::Moved => {
                if let Some(existing_touch) = self
                    .touch_state
                    .touches
                    .iter_mut()
                    .find(|t| t.id == touch.id)
                {
                    existing_touch.position = touch.position;
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touch_state.touches.retain(|t| t.id != touch.id);
            }
        }

        self.handle_touch_gestures();
        self.user_inptut = true;
    }

    fn handle_touch_gestures(&mut self) {
        let touch_count = self.touch_state.touches.len();

        match touch_count {
            1 => {
                // Single touch - camera rotation
                let touch = &self.touch_state.touches[0];
                if let Some(last_center) = self.touch_state.last_touch_center {
                    let dx = touch.position.0 - last_center.0;
                    let dy = touch.position.1 - last_center.1;

                    // Scale the touch movement similar to mouse movement but with better mobile sensitivity
                    self.rotation.x += dx * 0.3; // Reduced sensitivity for more precise control
                    self.rotation.y += dy * 0.3;
                }
                self.touch_state.last_touch_center = Some(touch.position);
            }
            2 => {
                // Two touches - pinch to zoom and pan
                let touch1 = &self.touch_state.touches[0];
                let touch2 = &self.touch_state.touches[1];

                let center_x = (touch1.position.0 + touch2.position.0) / 2.0;
                let center_y = (touch1.position.1 + touch2.position.1) / 2.0;
                let current_center = (center_x, center_y);

                // Calculate distance for pinch gesture
                let dx = touch2.position.0 - touch1.position.0;
                let dy = touch2.position.1 - touch1.position.1;
                let current_distance = (dx * dx + dy * dy).sqrt();

                if let Some(last_distance) = self.touch_state.last_pinch_distance {
                    // Pinch to zoom with improved sensitivity
                    let distance_change = current_distance - last_distance;
                    let zoom_factor = distance_change * 0.005; // Adjusted for better mobile zoom control
                    self.scroll += zoom_factor;
                }

                if let Some(last_center) = self.touch_state.last_touch_center {
                    // Pan with two fingers - improved sensitivity for mobile
                    let center_dx = current_center.0 - last_center.0;
                    let center_dy = current_center.1 - last_center.1;

                    self.shift.y += -center_dx * 0.3; // Reduced sensitivity for more precise panning
                    self.shift.x += center_dy * 0.3;
                }

                self.touch_state.last_pinch_distance = Some(current_distance);
                self.touch_state.last_touch_center = Some(current_center);
            }
            _ => {
                // No touches or more than 2 touches - reset state
                self.touch_state.last_pinch_distance = None;
                self.touch_state.last_touch_center = None;
            }
        }

        self.touch_state.last_touch_count = touch_count;
    }

    pub fn clear_touch_state(&mut self) {
        self.touch_state.touches.clear();
        self.touch_state.last_touch_count = 0;
        self.touch_state.last_pinch_distance = None;
        self.touch_state.last_touch_center = None;
    }

    /// moves the controller center to the closest point on a line defined by the camera position and rotation
    /// ajusts the controller up vector by projecting the current up vector onto the plane defined by the camera right vector
    pub fn reset_to_camera(&mut self, camera: PerspectiveCamera) {
        let inv_view = camera.rotation.invert();
        let forward = inv_view * Vector3::unit_z();
        let right = inv_view * Vector3::unit_x();

        // move center point
        self.center = closest_point(camera.position, forward, self.center);
        // adjust up vector by projecting it onto the plane defined by the right vector of the camera
        if let Some(up) = &self.up {
            let new_up = up - up.project_on(right);
            self.up = Some(new_up.normalize());
        }
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt: f32 = dt.as_secs_f32();
        let mut dir = camera.position - self.center;
        let distance = dir.magnitude();

        dir = dir.normalize_to((distance.ln() + self.scroll * dt * 10. * self.speed).exp());

        let view_t: Matrix3<f32> = camera.rotation.invert().into();

        let x_axis = view_t.x;
        let y_axis = self.up.unwrap_or(view_t.y);
        let z_axis = view_t.z;

        // Handle rotation
        let mut theta = Rad((self.rotation.x) * dt * self.sensitivity);
        let mut phi = Rad((-self.rotation.y) * dt * self.sensitivity);
        let mut eta = Rad::zero();

        if self.alt_pressed {
            eta = Rad(-self.rotation.y * dt * self.sensitivity);
            theta = Rad::zero();
            phi = Rad::zero();
        }

        let rot_theta = Quaternion::from_axis_angle(y_axis, theta);
        let rot_phi = Quaternion::from_axis_angle(x_axis, phi);
        let rot_eta = Quaternion::from_axis_angle(z_axis, eta);
        let rot = rot_theta * rot_phi * rot_eta;

        match self.mode {
            ControlMode::Orbit => {
                let offset = (self.shift.y * x_axis - self.shift.x * y_axis)
                    * dt
                    * self.speed
                    * 0.1
                    * distance;
                self.center += offset;
                camera.position += offset;

                let mut new_dir = rot.rotate_vector(dir);

                if angle_short(y_axis, new_dir) < Rad(0.1) {
                    new_dir = dir;
                }
                camera.position = self.center + new_dir;

                // update rotation
                camera.rotation = Quaternion::look_at(-new_dir, y_axis);
            }
            ControlMode::Fly => {
                // In Fly mode, we rotate the camera directly
                camera.rotation = camera.rotation * rot.invert();

                // Recalculate axes based on new rotation
                let view_t: Matrix3<f32> = camera.rotation.invert().into();
                let x_axis = view_t.x;
                let z_axis = view_t.z;

                // Movement is relative to camera view
                // amount.z is roughly forward/back (W/S)
                // amount.x is left/right (A/D)
                // amount.y is up/down (Space/Shift)

                // Note: The original 'amount' logic for Orbit assumes:
                // z += amount (W/UpperArrow), z -= amount (S/DownArrow)
                // x += amount (D/RightArrow), x -= amount (A/LeftArrow)
                // y += amount (Space), y -= amount (Shift)

                // In Orbit mode, specific movement wasn't implemented for amount.x/y/z directly into camera pos,
                // it was likely implicit or missing in the provided snippet beyond rotation/scroll.
                // Wait, checking process_keyboard again...
                // It updates self.amount.
                // But update_camera only uses self.shift?

                // Ah, I missed where self.amount is used in the original code.
                // Let me double check the original update_camera.
                // original code:
                // let offset = (self.shift.y * x_axis - self.shift.x * y_axis) * dt * self.speed * 0.1 * distance;
                // self.center += offset;
                // camera.position += offset;

                // It seems the original code IGNORED `self.amount` (keyboard input) for movement!
                // It only used `self.shift` which comes from right-click mouse drag.
                // Wait, let me check the original code again carefully.
                // Lines 86-125 update `self.amount`.
                // Lines 253-314 (update_camera) DO NOT USE `self.amount`.
                // This means WASD keys did NOTHING in the provided snippet?
                // Let me re-read the file content provided in Step 13.

                // Lines 86-125: process_keyboard updates `self.amount`.
                // Lines 253-314: update_camera uses `self.shift`, `self.rotation`, `self.scroll`.
                // It does NOT use `self.amount`.

                // This is strange. Maybe I missed it.
                // Let me check if `amount` is used elsewhere... No.
                // So WASD was effectively broken or unimplemented in the original code?
                // Or maybe `self.shift` is updated from `self.amount` somewhere?
                // No. `self.shift` is updated in `process_mouse` and `process_touch`.
                // `amount` is updated in `process_keyboard`.

                // I will implement Fly mode correctly using `self.amount`.
                // And I should probably fix Orbit mode to use `self.amount` too if it was intended,
                // but for now I will focus on Fly mode as requested.
                // Actually, if WASD didn't work, maybe the user wants it to work now?
                // The prompt says "implement a toggle...".

                let move_speed = dt * self.speed * 2.0 * distance.max(1.0); // improved speed scaling

                let forward = -z_axis;
                let right = x_axis;
                let up: Vector3<f32> = Vector3::unit_y(); // Global up, or camera up? Usually fly mode uses camera vectors.
                                                          // But for "Fly" usually W is forward in view direction.

                let mut move_vec: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);

                // Forward/Back
                move_vec += forward * self.amount.z;
                // Left/Right
                move_vec += right * self.amount.x;
                // Up/Down
                move_vec += Vector3::unit_y() * self.amount.y;

                camera.position += move_vec * move_speed;

                // Update center to match camera so switching back to orbit makes sense
                // We project center out in front of camera
                self.center = camera.position + forward * distance;
            }
        }

        // decay based on fps
        let mut decay = (0.8).powf(dt * 60.);
        if decay < 1e-4 {
            decay = 0.;
        }
        self.rotation *= decay;
        if self.rotation.magnitude() < 1e-4 {
            self.rotation = Vector3::zero();
        }
        self.shift *= decay;
        if self.shift.magnitude() < 1e-4 {
            self.shift = Vector2::zero();
        }
        self.scroll *= decay;
        if self.scroll.abs() < 1e-4 {
            self.scroll = 0.;
        }

        // Also decay amount for keyboard smoothness if desired, or reset it?
        // Usually keyboard input is continuous state (pressed vs not pressed).
        // process_keyboard sets amount to 1.0 or 0.0 or -1.0.
        // But the struct has `amount: Vector3<f32>`.
        // The process_keyboard adds/subtracts: `self.amount.z += amount`.
        // If I hold W, amount.z becomes 1.0. If I release, it becomes 0.0?
        // Wait, process_keyboard:
        // KeyCode::KeyW => self.amount.z += amount; (amount is 1.0 if pressed, 0.0 if released -- wait)
        // `let amount = if pressed { 1.0 } else { 0.0 };`
        // If processed pressed=true, amount=1.0, `self.amount.z += 1.0`.
        // If processed pressed=false, amount=0.0, `self.amount.z += 0.0`.
        // This implies `self.amount` accumulates indefinitely?
        // Ah, `ElementState::Released` logic in `lib.rs`:
        // `if event.state == ElementState::Released { ... }`
        // `state.controller.process_keyboard(key, event.state == ElementState::Pressed);`

        // If I press W: pressed=true, amount=1.0 => `self.amount.z += 1.0`.
        // If I release W: pressed=false, amount=0.0 => `self.amount.z += 0.0`.
        // THIS LOGIC IN process_keyboard SEEMS FLAWED if it's meant to be a state toggle.
        // It just increments `self.amount` when pressed, and does nothing when released.
        // Unless... `process_keyboard` is called differently?
        // Let's check `lib.rs` again.

        // `state.controller.process_keyboard(key, event.state == ElementState::Pressed)`

        // If pressed: amount=1. The code does `self.amount.z += 1.0`. Result: amount.z increases.
        // If released: amount=0. The code does `self.amount.z += 0.0`. Result: amount.z stays same.

        // THIS LOOKS LIKE A BUG in the existing code, explaining why WASD might not be working or used.
        // I should fix this to be a proper state tracking (Pressed = 1, Released = 0).
        // Or -1/0/1 logic.

        // I will fix `process_keyboard` to correctly set the state.

        self.user_inptut = false;
    }
}

fn closest_point(orig: Point3<f32>, dir: Vector3<f32>, point: Point3<f32>) -> Point3<f32> {
    let dir = dir.normalize();
    let lhs = point - orig;

    let dot_p = lhs.dot(dir);
    // Return result
    return orig + dir * dot_p;
}

fn angle_short(a: Vector3<f32>, b: Vector3<f32>) -> Rad<f32> {
    let angle = a.angle(b);
    if angle > Rad(PI / 2.) {
        return Rad(PI) - angle;
    } else {
        return angle;
    }
}

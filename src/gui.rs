#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "The reference rasterizer converts bounded window coordinates between screen numeric types"
)]

use ash::{Entry, vk};
use eyre::{Context, Result, bail};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::CString;
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const INITIAL_WIDTH: u32 = 1_200;
const INITIAL_HEIGHT: u32 = 760;
const BACKGROUND: Rgba = Rgba::new(0x00, 0x4d, 0x2a, 0xff);
const INK: Rgba = Rgba::new(0xdc, 0xe2, 0xdc, 0xff);
const ACTIVE: Rgba = Rgba::new(0xff, 0xb4, 0x5e, 0xff);
const INACTIVE: Rgba = Rgba::new(0x81, 0xa9, 0x91, 0xff);

/// Run the first native Teamy-Transcriber desktop surface.
///
/// The renderer is intentionally a small Ash/Vulkan transfer renderer. It
/// establishes the real window, surface, swapchain, resize, input, and redraw
/// lifecycle while the richer text and audio renderers remain replaceable.
///
/// # Errors
///
/// Returns an error when the event loop, Vulkan loader, window surface, or
/// presentation device cannot be initialized.
pub fn run() -> Result<()> {
    let event_loop = EventLoop::new().wrap_err("failed to create GUI event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut application = GuiApplication::default();
    event_loop
        .run_app(&mut application)
        .wrap_err("GUI event loop failed")
}

#[derive(Default)]
struct GuiApplication {
    window: Option<Window>,
    renderer: Option<VulkanRenderer>,
    state: GuiState,
}

impl ApplicationHandler for GuiApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Teamy-Transcriber")
            .with_inner_size(PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
            .with_min_inner_size(PhysicalSize::new(720, 520));
        let Ok(window) = event_loop.create_window(attributes) else {
            event_loop.exit();
            return;
        };

        match VulkanRenderer::new(&window) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
            }
            Err(error) => {
                eprintln!("failed to initialize Teamy-Transcriber GUI: {error:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    return;
                }
                if let (Some(window), Some(renderer)) =
                    (self.window.as_ref(), self.renderer.as_mut())
                    && let Err(error) = renderer.recreate_swapchain(window)
                {
                    eprintln!("failed to resize Teamy-Transcriber GUI: {error:#}");
                    event_loop.exit();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.state.cursor = position;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(window) = self.window.as_ref() {
                    self.state.click(window.inner_size());
                }
            }
            WindowEvent::RedrawRequested => self.draw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

impl GuiApplication {
    fn draw(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(renderer)) = (self.window.as_ref(), self.renderer.as_mut()) else {
            return;
        };
        self.state.phase += 0.025;
        match renderer.draw(&self.state) {
            Ok(true) => {
                if let Err(error) = renderer.recreate_swapchain(window) {
                    eprintln!("failed to recreate GUI swapchain: {error:#}");
                    event_loop.exit();
                }
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("failed to draw Teamy-Transcriber GUI: {error:#}");
                event_loop.exit();
            }
        }
    }
}

#[derive(Debug, Clone)]
struct GuiState {
    cursor: PhysicalPosition<f64>,
    phase: f32,
    recording: bool,
    transcript: String,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            cursor: PhysicalPosition::new(0.0, 0.0),
            phase: 0.0,
            recording: false,
            transcript: "Testing, 1, 2".to_string(),
        }
    }
}

impl GuiState {
    fn click(&mut self, size: PhysicalSize<u32>) {
        let width = size.width as f32;
        let height = size.height as f32;
        let mic_center = Point::new(width * 0.16, height * 0.38);
        let cursor = Point::new(self.cursor.x as f32, self.cursor.y as f32);
        if cursor.distance_squared(mic_center) <= (height * 0.115).powi(2) {
            self.recording = !self.recording;
            self.transcript = if self.recording {
                "Recording...".to_string()
            } else {
                "Testing, 1, 2".to_string()
            };
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Rgba {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Rgba {
    const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn distance_squared(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Debug)]
struct Canvas {
    width: u32,
    height: u32,
    bgra: bool,
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32, format: vk::Format) -> Result<Self> {
        let pixel_count = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| eyre::eyre!("GUI surface dimensions overflowed"))?;
        let bgra = matches!(
            format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
        );
        let mut canvas = Self {
            width,
            height,
            bgra,
            pixels: vec![0; pixel_count],
        };
        canvas.clear(BACKGROUND);
        Ok(canvas)
    }

    fn clear(&mut self, color: Rgba) {
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                self.set_pixel(x, y, color);
            }
        }
    }

    fn set_pixel(&mut self, x: i32, y: i32, color: Rgba) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let index = (y as usize * self.width as usize + x as usize) * 4;
        if self.bgra {
            self.pixels[index..index + 4].copy_from_slice(&[color.b, color.g, color.r, color.a]);
        } else {
            self.pixels[index..index + 4].copy_from_slice(&[color.r, color.g, color.b, color.a]);
        }
    }

    fn line(&mut self, start: Point, end: Point, color: Rgba) {
        let mut x0 = start.x.round() as i32;
        let mut y0 = start.y.round() as i32;
        let x1 = end.x.round() as i32;
        let y1 = end.y.round() as i32;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set_pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x0 += sx;
            }
            if twice <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn rect(&mut self, left: i32, top: i32, right: i32, bottom: i32, color: Rgba) {
        self.line(
            Point::new(left as f32, top as f32),
            Point::new(right as f32, top as f32),
            color,
        );
        self.line(
            Point::new(right as f32, top as f32),
            Point::new(right as f32, bottom as f32),
            color,
        );
        self.line(
            Point::new(right as f32, bottom as f32),
            Point::new(left as f32, bottom as f32),
            color,
        );
        self.line(
            Point::new(left as f32, bottom as f32),
            Point::new(left as f32, top as f32),
            color,
        );
    }

    fn circle(&mut self, center: Point, radius: f32, color: Rgba) {
        let steps = 96;
        let mut previous = Point::new(center.x + radius, center.y);
        for step in 1..=steps {
            let angle = std::f32::consts::TAU * step as f32 / steps as f32;
            let current = Point::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            self.line(previous, current, color);
            previous = current;
        }
    }

    fn draw_text(&mut self, origin: Point, text: &str, scale: i32, color: Rgba) {
        let mut x = origin.x as i32;
        let mut y = origin.y as i32;
        for character in text.chars() {
            if character == '\n' {
                x = origin.x as i32;
                y += 8 * scale;
                continue;
            }
            let glyph = glyph(character);
            for (row, bits) in glyph.iter().copied().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                self.set_pixel(
                                    x + column * scale + dx,
                                    y + row as i32 * scale + dy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            x += 6 * scale;
        }
    }

    fn panel(&mut self, left: i32, top: i32, right: i32, bottom: i32, color: Rgba) {
        self.rect(left, top, right, bottom, color);
        self.rect(left + 2, top + 2, right - 2, bottom - 2, color);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The reference layout is kept together so the first GUI slice mirrors the supplied sketch"
    )]
    fn render_ui(&mut self, state: &GuiState) {
        self.clear(BACKGROUND);
        let width = self.width as f32;
        let height = self.height as f32;
        let margin = (width * 0.018).max(16.0) as i32;
        let ink = if state.recording { ACTIVE } else { INK };

        self.panel(margin, 16, self.width as i32 - margin, 94, ink);
        self.draw_text(
            Point::new((margin + 24) as f32, 35.0),
            "TEAMY-TRANSCRIBER",
            4,
            ink,
        );
        self.draw_text(
            Point::new((self.width as i32 - 220) as f32, 46.0),
            if state.recording {
                "RECORDING"
            } else {
                "READY"
            },
            2,
            if state.recording { ACTIVE } else { INACTIVE },
        );

        let mic_center = Point::new(width * 0.16, height * 0.38);
        let mic_radius = (height * 0.115).max(58.0);
        self.circle(mic_center, mic_radius, ink);
        self.circle(mic_center, mic_radius - 3.0, ink);
        self.circle(Point::new(mic_center.x, mic_center.y - 12.0), 19.0, ink);
        self.line(
            Point::new(mic_center.x - 19.0, mic_center.y - 12.0),
            Point::new(mic_center.x - 19.0, mic_center.y + 18.0),
            ink,
        );
        self.line(
            Point::new(mic_center.x + 19.0, mic_center.y - 12.0),
            Point::new(mic_center.x + 19.0, mic_center.y + 18.0),
            ink,
        );
        self.line(
            Point::new(mic_center.x - 19.0, mic_center.y + 18.0),
            Point::new(mic_center.x, mic_center.y + 30.0),
            ink,
        );
        self.line(
            Point::new(mic_center.x + 19.0, mic_center.y + 18.0),
            Point::new(mic_center.x, mic_center.y + 30.0),
            ink,
        );
        self.line(
            Point::new(mic_center.x, mic_center.y + 30.0),
            Point::new(mic_center.x, mic_center.y + 48.0),
            ink,
        );
        self.line(
            Point::new(mic_center.x - 24.0, mic_center.y + 49.0),
            Point::new(mic_center.x + 24.0, mic_center.y + 49.0),
            ink,
        );

        let selector_left = (width * 0.27) as i32;
        let selector_right = (width * 0.63) as i32;
        self.panel(
            selector_left,
            (height * 0.29) as i32,
            selector_right,
            (height * 0.38) as i32,
            ink,
        );
        self.panel(
            selector_left,
            (height * 0.40) as i32,
            selector_right,
            (height * 0.49) as i32,
            ink,
        );
        self.draw_text(
            Point::new((selector_left + 48) as f32, height * 0.315),
            "MICROPHONE: WOER",
            3,
            ink,
        );
        self.draw_text(
            Point::new((selector_left + 48) as f32, height * 0.425),
            "SAVE DIR: ~/DOWNLOADS",
            3,
            ink,
        );

        let panel_left = (width * 0.06) as i32;
        let panel_right = (width * 0.67) as i32;
        let waveform_top = (height * 0.55) as i32;
        let waveform_bottom = (height * 0.72) as i32;
        let transcript_top = (height * 0.75) as i32;
        self.panel(panel_left, waveform_top, panel_right, waveform_bottom, ink);
        self.panel(
            panel_left,
            transcript_top,
            panel_right,
            height as i32 - 34,
            ink,
        );

        let center_y = (waveform_top + waveform_bottom) as f32 * 0.5;
        let amplitude = (waveform_bottom - waveform_top) as f32 * 0.34;
        let mut previous = Point::new(panel_left as f32 + 12.0, center_y);
        for index in 1..=180 {
            let fraction = index as f32 / 180.0;
            let x = panel_left as f32 + 12.0 + fraction * (panel_right - panel_left - 24) as f32;
            let y = center_y
                + (fraction * 32.0 + state.phase).sin()
                    * amplitude
                    * (0.45 + 0.55 * (fraction * 11.0).sin().abs());
            let current = Point::new(x, y);
            self.line(previous, current, ink);
            previous = current;
        }
        self.draw_text(
            Point::new((panel_left + 30) as f32, (transcript_top + 54) as f32),
            &state.transcript,
            4,
            ink,
        );
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Keeping the small fixed bitmap alphabet together makes the renderer deterministic"
)]
fn glyph(character: char) -> [u8; 7] {
    match character.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        ':' => [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0],
        '~' => [0, 0, 0b01001, 0b10110, 0, 0, 0],
        ',' => [0, 0, 0, 0, 0, 0b00100, 0b01000],
        _ => [0; 7],
    }
}

struct VulkanRenderer {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    initialized_images: Vec<bool>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
}

impl VulkanRenderer {
    #[expect(
        clippy::too_many_lines,
        reason = "Vulkan initialization is kept as one ordered transaction so cleanup ownership is auditable"
    )]
    fn new(window: &Window) -> Result<Self> {
        // SAFETY: Loading the process Vulkan loader is the first step before any
        // instance/device handle is used.
        let entry = unsafe { Entry::load().wrap_err("failed to load Vulkan loader")? };
        let app_name = CString::new("teamy-transcriber").expect("static app name has no NUL");
        let engine_name = CString::new("teamy-transcriber").expect("static engine name has no NUL");
        let display_handle = window
            .display_handle()
            .wrap_err("failed to obtain display handle")?
            .as_raw();
        let extensions = ash_window::enumerate_required_extensions(display_handle)
            .wrap_err("failed to enumerate required window extensions")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extensions);
        // SAFETY: The instance create-info references live application names and
        // only the extensions reported by the windowing backend.
        let instance = unsafe {
            entry
                .create_instance(&instance_info, None)
                .wrap_err("failed to create Vulkan instance")?
        };
        let window_handle = window
            .window_handle()
            .wrap_err("failed to obtain window handle")?
            .as_raw();
        // SAFETY: The raw display/window handles belong to the live Winit window
        // and remain valid for the renderer lifetime.
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .wrap_err("failed to create Vulkan window surface")?
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let (physical_device, queue_family_index) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        // SAFETY: The selected queue family supports graphics and presentation,
        // and the swapchain extension is enabled for the logical device.
        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .wrap_err("failed to create Vulkan logical device")?
        };
        // SAFETY: Queue zero exists in the selected queue family created above.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, format, extent, images) = create_swapchain(
            window,
            &surface_loader,
            surface,
            &swapchain_loader,
            physical_device,
            vk::SwapchainKHR::null(),
        )?;
        let image_count = images.len();
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: The command pool uses the selected live graphics queue family.
        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .wrap_err("failed to create GUI command pool")?
        };
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The command buffer allocation references the live command pool.
        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_info)
                .wrap_err("failed to allocate GUI command buffer")?
        }[0];
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // SAFETY: The semaphore create-info is complete and the device is live.
        let image_available = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .wrap_err("failed to create image-available semaphore")?
        };
        // SAFETY: The semaphore create-info is complete and the device is live.
        let render_finished = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .wrap_err("failed to create render-finished semaphore")?
        };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: The fence create-info is complete and the device is live.
        let fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .wrap_err("failed to create GUI fence")?
        };
        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            swapchain_loader,
            swapchain,
            format,
            extent,
            images,
            initialized_images: vec![false; image_count],
            command_pool,
            command_buffer,
            fence,
            image_available,
            render_finished,
        })
    }

    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // SAFETY: The device is live and waiting makes swapchain destruction safe.
        unsafe { self.device.device_wait_idle() }
            .wrap_err("failed waiting for GUI device during resize")?;
        let old_swapchain = self.swapchain;
        let (swapchain, format, extent, images) = create_swapchain(
            window,
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            self.physical_device,
            old_swapchain,
        )?;
        // SAFETY: The old swapchain is no longer in use after device_wait_idle.
        unsafe { self.swapchain_loader.destroy_swapchain(old_swapchain, None) };
        self.swapchain = swapchain;
        self.format = format;
        self.extent = extent;
        self.images = images;
        self.initialized_images = vec![false; self.images.len()];
        Ok(())
    }

    fn draw(&mut self, state: &GuiState) -> Result<bool> {
        // SAFETY: The fence belongs to this live device and is used for this frame.
        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .wrap_err("failed waiting for GUI frame fence")?;
        // SAFETY: The swapchain and image-available semaphore belong to the live device.
        let (image_index, suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
            Err(error) => return Err(error).wrap_err("failed to acquire GUI swapchain image"),
        };
        let image_index = usize::try_from(image_index).wrap_err("invalid swapchain image index")?;
        // SAFETY: The previous frame was waited above, so these frame resources
        // are not in flight.
        unsafe { self.device.reset_fences(&[self.fence]) }
            .wrap_err("failed to reset GUI frame fence")?;
        // SAFETY: The command buffer is allocated from a resettable command pool
        // and the previous submission has completed.
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .wrap_err("failed to reset GUI command buffer")?;
        let mut canvas = Canvas::new(self.extent.width, self.extent.height, self.format)?;
        canvas.render_ui(state);
        let (staging_buffer, staging_memory) = self.create_staging_buffer(&canvas.pixels)?;
        let command_result =
            self.record_copy(image_index, staging_buffer, canvas.width, canvas.height);
        if let Err(error) = command_result {
            // SAFETY: Recording failed before submission, so the staging resources
            // are not referenced by the device.
            unsafe { self.device.destroy_buffer(staging_buffer, None) };
            // SAFETY: The staging allocation is no longer bound to a live buffer.
            unsafe { self.device.free_memory(staging_memory, None) };
            return Err(error);
        }
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&self.image_available))
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&self.command_buffer))
            .signal_semaphores(std::slice::from_ref(&self.render_finished));
        // SAFETY: The command buffer, semaphores, queue, and fence all belong to
        // this live device and the wait/signal chain is fully specified.
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], self.fence)
        }
        .wrap_err("failed to submit GUI frame")?;
        let present_image_index = image_index as u32;
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&self.render_finished))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(std::slice::from_ref(&present_image_index));
        // SAFETY: The swapchain image was acquired above and the render-finished
        // semaphore is signaled by the submitted command buffer.
        let present_suboptimal = match unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        } {
            Ok(value) => value,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(error) => return Err(error).wrap_err("failed to present GUI frame"),
        };
        // SAFETY: Waiting for idle completes the copy before its staging resources
        // are released.
        unsafe { self.device.device_wait_idle() }
            .wrap_err("failed waiting for GUI frame completion")?;
        // SAFETY: The device is idle and no command references the staging buffer.
        unsafe { self.device.destroy_buffer(staging_buffer, None) };
        // SAFETY: The device is idle and the staging buffer has been destroyed.
        unsafe { self.device.free_memory(staging_memory, None) };
        self.initialized_images[image_index] = true;
        Ok(suboptimal || present_suboptimal)
    }

    fn record_copy(
        &mut self,
        image_index: usize,
        staging_buffer: vk::Buffer,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let old_layout = if self.initialized_images[image_index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let to_transfer = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(self.images[image_index])
            .subresource_range(subresource_range);
        let to_present = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .image(self.images[image_index])
            .subresource_range(subresource_range);
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: The command buffer was reset above and all referenced handles
        // belong to this live device.
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
        }
        .wrap_err("failed to begin GUI command buffer")?;
        // SAFETY: The barrier transitions the acquired swapchain image from its
        // tracked layout to the transfer destination layout.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );
        };
        // SAFETY: The staging buffer contains exactly width*height RGBA/BGRA
        // bytes and the destination image has the matching swapchain extent.
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging_buffer,
                self.images[image_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        };
        // SAFETY: The final barrier makes the copied image available for present.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        };
        // SAFETY: All commands in this recording use live handles and valid
        // command-buffer state.
        unsafe { self.device.end_command_buffer(self.command_buffer) }
            .wrap_err("failed to record GUI command buffer")?;
        Ok(())
    }

    fn create_staging_buffer(&self, pixels: &[u8]) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let size =
            vk::DeviceSize::try_from(pixels.len()).wrap_err("GUI pixel buffer is too large")?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: The buffer create-info is complete and the device is live.
        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .wrap_err("failed to create GUI staging buffer")?
        };
        // SAFETY: The buffer was created on this live device.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            &self.instance,
            self.physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: The allocation request uses requirements returned for the live
        // staging buffer and a compatible host-visible memory type.
        let memory = unsafe {
            self.device
                .allocate_memory(&allocate_info, None)
                .wrap_err("failed to allocate GUI staging memory")?
        };
        // SAFETY: The allocation was selected for this buffer's memory
        // requirements and has not been bound elsewhere.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .wrap_err("failed to bind GUI staging memory")?;
        // SAFETY: The host-visible allocation is large enough for the pixel slice
        // and remains mapped only for the copy below.
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .wrap_err("failed to map GUI staging memory")?;
        // SAFETY: `mapped` points to at least pixels.len() writable bytes, and the
        // source and destination ranges do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped.cast::<u8>(), pixels.len());
        };
        // SAFETY: The mapped range belongs to this live allocation and the copy is
        // complete before it is unmapped.
        unsafe { self.device.unmap_memory(memory) };
        Ok((buffer, memory))
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // SAFETY: Drop is the final owner of the live device resources, so idle
        // guarantees no submitted work references the objects being destroyed.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        // SAFETY: The device is idle and owns this semaphore.
        unsafe { self.device.destroy_semaphore(self.render_finished, None) };
        // SAFETY: The device is idle and owns this semaphore.
        unsafe { self.device.destroy_semaphore(self.image_available, None) };
        // SAFETY: The device is idle and owns this fence.
        unsafe { self.device.destroy_fence(self.fence, None) };
        // SAFETY: The device is idle and owns this command pool.
        unsafe { self.device.destroy_command_pool(self.command_pool, None) };
        // SAFETY: The device is idle and owns this swapchain.
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        };
        // SAFETY: All child device resources have been destroyed first.
        unsafe { self.device.destroy_device(None) };
        // SAFETY: The logical device is gone and the instance still owns this surface.
        unsafe { self.surface_loader.destroy_surface(self.surface, None) };
        // SAFETY: The instance is the final Vulkan owner and no child handles remain.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    // SAFETY: The Vulkan instance is live for the duration of this query.
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .wrap_err("failed to enumerate Vulkan physical devices")?
    };
    for physical_device in physical_devices {
        // SAFETY: The physical device was returned by the live instance.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in queue_families.iter().enumerate() {
            let index = u32::try_from(index).wrap_err("queue family index overflowed")?;
            // SAFETY: The surface and physical device belong to the live instance.
            let present_support = unsafe {
                surface_loader
                    .get_physical_device_surface_support(physical_device, index, surface)
                    .wrap_err("failed to query surface support")?
            };
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present_support {
                return Ok((physical_device, index));
            }
        }
    }
    bail!("no Vulkan physical device supports graphics presentation")
}

fn create_swapchain(
    window: &Window,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    old_swapchain: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
    // SAFETY: The surface and physical device belong to the live instance.
    let capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .wrap_err("failed to query GUI surface capabilities")?
    };
    if !capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        bail!("GUI surface does not support transfer-to-present images");
    }
    // SAFETY: The surface and physical device belong to the live instance.
    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .wrap_err("failed to query GUI surface formats")?
    };
    let format = formats
        .iter()
        .copied()
        .find(|format| {
            matches!(
                format.format,
                vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
            )
        })
        .or_else(|| formats.first().copied())
        .ok_or_else(|| eyre::eyre!("GUI surface reported no supported formats"))?;
    let window_size = window.inner_size();
    let extent = if capabilities.current_extent.width == u32::MAX {
        vk::Extent2D {
            width: window_size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    } else {
        capabilities.current_extent
    };
    let mut image_count = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count != 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(choose_composite_alpha(
            capabilities.supported_composite_alpha,
        ))
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old_swapchain);
    // SAFETY: The create-info references live surface/device state and valid
    // presentation capabilities.
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .wrap_err("failed to create GUI swapchain")?
    };
    // SAFETY: The swapchain was created successfully on the live device.
    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .wrap_err("failed to enumerate GUI swapchain images")?
    };
    Ok((swapchain, format.format, extent, images))
}

fn choose_composite_alpha(supported: vk::CompositeAlphaFlagsKHR) -> vk::CompositeAlphaFlagsKHR {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .into_iter()
    .find(|candidate| supported.contains(*candidate))
    .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    // SAFETY: The physical device was returned by the live instance.
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for index in 0..properties.memory_type_count {
        let supported = type_filter & (1 << index) != 0;
        let memory_type = properties.memory_types[index as usize];
        if supported && memory_type.property_flags.contains(required) {
            return Ok(index);
        }
    }
    bail!("no Vulkan memory type satisfies GUI staging requirements")
}

#[cfg(test)]
mod tests {
    use super::{GuiState, INITIAL_HEIGHT, INITIAL_WIDTH, glyph};
    use winit::dpi::{PhysicalPosition, PhysicalSize};

    #[test]
    fn reference_labels_have_bitmap_glyphs() {
        for character in "TEAMY-TRANSCRIBER MICROPHONE: WOER SAVE DIR: ~/DOWNLOADS TESTING, 1, 2"
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
        {
            assert!(
                glyph(character).iter().any(|row| *row != 0),
                "missing glyph for {character:?}"
            );
        }
    }

    #[test]
    fn microphone_hit_toggles_recording_state() {
        let size = PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT);
        let mut state = GuiState {
            cursor: PhysicalPosition::new(192.0, 288.8),
            ..GuiState::default()
        };

        state.click(size);
        assert!(state.recording);
        assert_eq!(state.transcript, "Recording...");

        state.click(size);
        assert!(!state.recording);
        assert_eq!(state.transcript, "Testing, 1, 2");
    }
}

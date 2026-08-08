#![allow(
    clippy::borrow_as_ptr,
    clippy::default_trait_access,
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks,
    reason = "this module is the narrowly scoped Win32 FFI boundary for the tray"
)]

use super::GuiMessage;
use super::TrayAction;
use eyre::Context;
use eyre::ContextCompat;
use eyre::Result;
use eyre::bail;
use std::ffi::c_void;
use std::sync::mpsc::Sender;
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::sync_channel;
use std::thread;
use std::thread::JoinHandle;
use windows::Win32::Foundation::HWND;
use windows::Win32::Foundation::LPARAM;
use windows::Win32::Foundation::LRESULT;
use windows::Win32::Foundation::POINT;
use windows::Win32::Foundation::WPARAM;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::HOT_KEY_MODIFIERS;
use windows::Win32::UI::Input::KeyboardAndMouse::MOD_CONTROL;
use windows::Win32::UI::Input::KeyboardAndMouse::MOD_SHIFT;
use windows::Win32::UI::Input::KeyboardAndMouse::RegisterHotKey;
use windows::Win32::UI::Input::KeyboardAndMouse::UnregisterHotKey;
use windows::Win32::UI::Input::KeyboardAndMouse::VK_SPACE;
use windows::Win32::UI::Shell::NIF_ICON;
use windows::Win32::UI::Shell::NIF_MESSAGE;
use windows::Win32::UI::Shell::NIF_TIP;
use windows::Win32::UI::Shell::NIM_ADD;
use windows::Win32::UI::Shell::NIM_DELETE;
use windows::Win32::UI::Shell::NOTIFYICONDATAW;
use windows::Win32::UI::Shell::Shell_NotifyIconW;
use windows::Win32::UI::WindowsAndMessaging::AppendMenuW;
use windows::Win32::UI::WindowsAndMessaging::CREATESTRUCTW;
use windows::Win32::UI::WindowsAndMessaging::CreatePopupMenu;
use windows::Win32::UI::WindowsAndMessaging::CreateWindowExW;
use windows::Win32::UI::WindowsAndMessaging::DefWindowProcW;
use windows::Win32::UI::WindowsAndMessaging::DestroyMenu;
use windows::Win32::UI::WindowsAndMessaging::DestroyWindow;
use windows::Win32::UI::WindowsAndMessaging::DispatchMessageW;
use windows::Win32::UI::WindowsAndMessaging::GetCursorPos;
use windows::Win32::UI::WindowsAndMessaging::GetMessageW;
use windows::Win32::UI::WindowsAndMessaging::GetWindowLongPtrW;
use windows::Win32::UI::WindowsAndMessaging::HICON;
use windows::Win32::UI::WindowsAndMessaging::LoadIconW;
use windows::Win32::UI::WindowsAndMessaging::MF_SEPARATOR;
use windows::Win32::UI::WindowsAndMessaging::MF_STRING;
use windows::Win32::UI::WindowsAndMessaging::MSG;
use windows::Win32::UI::WindowsAndMessaging::PostMessageW;
use windows::Win32::UI::WindowsAndMessaging::PostQuitMessage;
use windows::Win32::UI::WindowsAndMessaging::RegisterClassW;
use windows::Win32::UI::WindowsAndMessaging::RegisterWindowMessageW;
use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
use windows::Win32::UI::WindowsAndMessaging::SetWindowLongPtrW;
use windows::Win32::UI::WindowsAndMessaging::TPM_LEFTALIGN;
use windows::Win32::UI::WindowsAndMessaging::TPM_RETURNCMD;
use windows::Win32::UI::WindowsAndMessaging::TPM_RIGHTBUTTON;
use windows::Win32::UI::WindowsAndMessaging::TPM_TOPALIGN;
use windows::Win32::UI::WindowsAndMessaging::TrackPopupMenu;
use windows::Win32::UI::WindowsAndMessaging::TranslateMessage;
use windows::Win32::UI::WindowsAndMessaging::WM_CLOSE;
use windows::Win32::UI::WindowsAndMessaging::WM_CREATE;
use windows::Win32::UI::WindowsAndMessaging::WM_DESTROY;
use windows::Win32::UI::WindowsAndMessaging::WM_HOTKEY;
use windows::Win32::UI::WindowsAndMessaging::WM_LBUTTONDBLCLK;
use windows::Win32::UI::WindowsAndMessaging::WM_NCCREATE;
use windows::Win32::UI::WindowsAndMessaging::WM_RBUTTONUP;
use windows::Win32::UI::WindowsAndMessaging::WM_USER;
use windows::Win32::UI::WindowsAndMessaging::WNDCLASSW;
use windows::Win32::UI::WindowsAndMessaging::WS_OVERLAPPEDWINDOW;
use windows::core::w;

const HOTKEY_ID: i32 = 1;
const TRAY_ICON_ID: u32 = 1;
const WM_TRAY_CALLBACK: u32 = WM_USER + 1;
const WM_TRAY_COMMAND: u32 = WM_USER + 2;
const CMD_SHOW_WINDOW: usize = 0x5000;
const CMD_TOGGLE_RECORDING: usize = 0x5001;
const CMD_TOGGLE_HOTKEY: usize = 0x5002;
const CMD_EXIT: usize = 0x5003;

#[derive(Debug)]
pub(crate) struct TrayController {
    hwnd: isize,
    thread: Option<JoinHandle<()>>,
}

impl TrayController {
    pub(crate) fn start(sender: Sender<GuiMessage>, hotkey_enabled: bool) -> Result<Self> {
        let (ready_sender, ready_receiver) = sync_channel(1);
        let thread = thread::Builder::new()
            .name("teamy-transcriber-tray".to_string())
            .spawn(move || {
                if let Err(error) = run_tray(sender, hotkey_enabled, &ready_sender) {
                    let _ = ready_sender.send(Err(error.to_string()));
                }
            })
            .wrap_err("failed to start the tray thread")?;
        match ready_receiver.recv() {
            Ok(Ok(hwnd)) => Ok(Self {
                hwnd,
                thread: Some(thread),
            }),
            Ok(Err(message)) => {
                let _ = thread.join();
                bail!("tray initialization failed: {message}");
            }
            Err(error) => {
                let _ = thread.join();
                bail!("tray initialization ended before reporting readiness: {error}");
            }
        }
    }

    pub(crate) fn set_hotkey_enabled(&self, enabled: bool) {
        let _ = unsafe {
            PostMessageW(
                Some(hwnd_from_bits(self.hwnd)),
                WM_TRAY_COMMAND,
                WPARAM(CMD_TOGGLE_HOTKEY),
                LPARAM(isize::from(enabled)),
            )
        };
    }
}

impl Drop for TrayController {
    fn drop(&mut self) {
        let _ = unsafe {
            PostMessageW(
                Some(hwnd_from_bits(self.hwnd)),
                WM_CLOSE,
                WPARAM(0),
                LPARAM(0),
            )
        };
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Debug)]
struct TrayState {
    sender: Sender<GuiMessage>,
    hotkey_enabled: bool,
    taskbar_created_message: u32,
}

fn run_tray(
    sender: Sender<GuiMessage>,
    hotkey_enabled: bool,
    ready_sender: &SyncSender<std::result::Result<isize, String>>,
) -> Result<()> {
    let hinstance = unsafe { GetModuleHandleW(None) }.wrap_err("GetModuleHandleW failed")?;
    let class_name = w!("teamy_transcriber_tray_window");
    let taskbar_created_message = unsafe { RegisterWindowMessageW(w!("TaskbarCreated")) };
    let wnd_class = WNDCLASSW {
        lpfnWndProc: Some(window_proc),
        hInstance: hinstance.into(),
        lpszClassName: class_name,
        ..Default::default()
    };
    let atom = unsafe { RegisterClassW(&wnd_class) };
    if atom == 0 {
        bail!("RegisterClassW failed for the tray window");
    }

    let state = Box::new(TrayState {
        sender,
        hotkey_enabled: false,
        taskbar_created_message,
    });
    let state_pointer = Box::into_raw(state);
    let hwnd = match unsafe {
        CreateWindowExW(
            Default::default(),
            class_name,
            w!("Teamy-Transcriber tray"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            0,
            0,
            None,
            None,
            Some(hinstance.into()),
            Some(state_pointer.cast()),
        )
    } {
        Ok(hwnd) => hwnd,
        Err(error) => {
            unsafe { drop(Box::from_raw(state_pointer)) };
            return Err(error).wrap_err("CreateWindowExW failed for the tray window");
        }
    };

    if let Err(error) = add_tray_icon(hwnd) {
        let _ = unsafe { DestroyWindow(hwnd) };
        return Err(error);
    }
    with_state(hwnd, |state| {
        let warning = state.set_hotkey(hwnd, hotkey_enabled);
        state.report_hotkey_state(warning);
    });
    ready_sender
        .send(Ok(hwnd.0 as isize))
        .map_err(|error| eyre::eyre!("failed to report tray readiness: {error}"))?;
    run_message_loop()
}

fn run_message_loop() -> Result<()> {
    let mut message = MSG::default();
    loop {
        let result = unsafe { GetMessageW(&mut message, None, 0, 0) };
        if result.0 == -1 {
            bail!("GetMessageW failed while running the tray message loop");
        }
        if result.0 == 0 {
            break;
        }
        let _ = unsafe { TranslateMessage(&message) };
        unsafe { DispatchMessageW(&message) };
    }
    Ok(())
}

impl TrayState {
    fn set_hotkey(&mut self, hwnd: HWND, enabled: bool) -> Option<String> {
        if enabled == self.hotkey_enabled {
            return None;
        }
        if enabled {
            if let Err(error) = register_hotkey(hwnd) {
                return Some(format!("global hotkey unavailable: {error}"));
            }
        } else {
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_ID) };
        }
        self.hotkey_enabled = enabled;
        None
    }

    fn toggle_hotkey(&mut self, hwnd: HWND) {
        let next = !self.hotkey_enabled;
        let warning = self.set_hotkey(hwnd, next);
        self.report_hotkey_state(warning);
    }

    fn report_hotkey_state(&self, warning: Option<String>) {
        let _ = self
            .sender
            .send(GuiMessage::TrayHotkeyChanged(self.hotkey_enabled));
        if let Some(message) = warning {
            let _ = self.sender.send(GuiMessage::TrayStatus { message });
        }
    }

    fn send_action(&self, action: TrayAction) {
        let _ = self.sender.send(GuiMessage::Tray(action));
    }
}

fn register_hotkey(hwnd: HWND) -> Result<()> {
    let modifiers: HOT_KEY_MODIFIERS = MOD_CONTROL | MOD_SHIFT;
    unsafe { RegisterHotKey(Some(hwnd), HOTKEY_ID, modifiers, u32::from(VK_SPACE.0)) }
        .ok()
        .wrap_err("Ctrl+Shift+Space may already be registered by another application")
}

fn add_tray_icon(hwnd: HWND) -> Result<()> {
    let icon = load_tray_icon()?;
    let data = notify_data(hwnd, icon);
    unsafe { Shell_NotifyIconW(NIM_ADD, &data).ok() }
        .wrap_err("failed to add the Teamy-Transcriber tray icon")
}

fn re_add_tray_icon(hwnd: HWND) -> Result<()> {
    let icon = load_tray_icon()?;
    let data = notify_data(hwnd, icon);
    unsafe { Shell_NotifyIconW(NIM_ADD, &data).ok() }
        .wrap_err("failed to restore the Teamy-Transcriber tray icon")
}

fn delete_tray_icon(hwnd: HWND) -> Result<()> {
    let data = NOTIFYICONDATAW {
        cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
        hWnd: hwnd,
        uID: TRAY_ICON_ID,
        ..Default::default()
    };
    unsafe { Shell_NotifyIconW(NIM_DELETE, &data).ok() }
        .wrap_err("failed to remove the Teamy-Transcriber tray icon")
}

fn load_tray_icon() -> Result<HICON> {
    let module = unsafe { GetModuleHandleW(None) }.wrap_err("GetModuleHandleW failed")?;
    unsafe { LoadIconW(Some(module.into()), w!("main_icon")) }
        .wrap_err("failed to load the embedded Teamy-Transcriber tray icon")
}

fn notify_data(hwnd: HWND, icon: HICON) -> NOTIFYICONDATAW {
    let mut data = NOTIFYICONDATAW {
        cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
        hWnd: hwnd,
        uID: TRAY_ICON_ID,
        uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
        uCallbackMessage: WM_TRAY_CALLBACK,
        hIcon: icon,
        ..Default::default()
    };
    let tip: Vec<u16> = "Teamy-Transcriber".encode_utf16().chain(Some(0)).collect();
    let length = tip.len().min(data.szTip.len());
    data.szTip[..length].copy_from_slice(&tip[..length]);
    data
}

fn show_context_menu(hwnd: HWND) {
    let Some(menu) = (unsafe { CreatePopupMenu() }).ok() else {
        return;
    };
    unsafe {
        let _ = AppendMenuW(menu, MF_STRING, CMD_SHOW_WINDOW, w!("Show window"));
        let _ = AppendMenuW(
            menu,
            MF_STRING,
            CMD_TOGGLE_RECORDING,
            w!("Start or stop recording"),
        );
        let _ = AppendMenuW(
            menu,
            MF_STRING,
            CMD_TOGGLE_HOTKEY,
            w!("Toggle Ctrl+Shift+Space hotkey"),
        );
        let _ = AppendMenuW(menu, MF_SEPARATOR, 0, windows::core::PCWSTR::null());
        let _ = AppendMenuW(menu, MF_STRING, CMD_EXIT, w!("Exit"));
        let _ = SetForegroundWindow(hwnd);
    }
    let mut cursor = POINT::default();
    let _ = unsafe { GetCursorPos(&mut cursor) };
    let selection = unsafe {
        TrackPopupMenu(
            menu,
            TPM_RIGHTBUTTON | TPM_TOPALIGN | TPM_LEFTALIGN | TPM_RETURNCMD,
            cursor.x,
            cursor.y,
            Some(0),
            hwnd,
            None,
        )
    };
    let _ = unsafe { DestroyMenu(menu) };
    if selection.0 == 0 {
        return;
    }
    with_state(hwnd, |state| match selection.0 as usize {
        CMD_SHOW_WINDOW => state.send_action(TrayAction::ShowWindow),
        CMD_TOGGLE_RECORDING => state.send_action(TrayAction::ToggleRecording),
        CMD_TOGGLE_HOTKEY => state.toggle_hotkey(hwnd),
        CMD_EXIT => state.send_action(TrayAction::Exit),
        _ => {}
    });
}

fn with_state(hwnd: HWND, action: impl FnOnce(&mut TrayState)) {
    let pointer =
        unsafe { GetWindowLongPtrW(hwnd, windows::Win32::UI::WindowsAndMessaging::GWLP_USERDATA) };
    if pointer == 0 {
        return;
    }
    let state = unsafe { &mut *(pointer as *mut TrayState) };
    action(state);
}

fn hwnd_from_bits(bits: isize) -> HWND {
    HWND(bits as *mut c_void)
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match message {
        WM_NCCREATE => {
            let create = unsafe { &*(lparam.0 as *const CREATESTRUCTW) };
            unsafe {
                SetWindowLongPtrW(
                    hwnd,
                    windows::Win32::UI::WindowsAndMessaging::GWLP_USERDATA,
                    create.lpCreateParams as isize,
                )
            };
            LRESULT(1)
        }
        WM_CREATE => LRESULT(0),
        WM_HOTKEY if wparam.0 as i32 == HOTKEY_ID => {
            with_state(hwnd, |state| state.send_action(TrayAction::ToggleRecording));
            LRESULT(0)
        }
        WM_TRAY_CALLBACK => {
            match lparam.0 as u32 {
                WM_RBUTTONUP => show_context_menu(hwnd),
                WM_LBUTTONDBLCLK => {
                    with_state(hwnd, |state| state.send_action(TrayAction::ShowWindow));
                }
                _ => {}
            }
            LRESULT(0)
        }
        WM_TRAY_COMMAND if wparam.0 == CMD_TOGGLE_HOTKEY => {
            with_state(hwnd, |state| {
                let warning = state.set_hotkey(hwnd, lparam.0 != 0);
                state.report_hotkey_state(warning);
            });
            LRESULT(0)
        }
        WM_CLOSE => {
            let _ = unsafe { DestroyWindow(hwnd) };
            LRESULT(0)
        }
        WM_DESTROY => {
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_ID) };
            let _ = delete_tray_icon(hwnd);
            let pointer = unsafe {
                SetWindowLongPtrW(
                    hwnd,
                    windows::Win32::UI::WindowsAndMessaging::GWLP_USERDATA,
                    0,
                )
            };
            if pointer != 0 {
                unsafe { drop(Box::from_raw(pointer as *mut TrayState)) };
            }
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        _ => {
            let taskbar_created = with_taskbar_message(hwnd);
            if taskbar_created != 0 && message == taskbar_created {
                let _ = re_add_tray_icon(hwnd);
                LRESULT(0)
            } else {
                unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
            }
        }
    }
}

fn with_taskbar_message(hwnd: HWND) -> u32 {
    let pointer =
        unsafe { GetWindowLongPtrW(hwnd, windows::Win32::UI::WindowsAndMessaging::GWLP_USERDATA) };
    if pointer == 0 {
        return 0;
    }
    unsafe { (*(pointer as *const TrayState)).taskbar_created_message }
}

use facet::Facet;
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct AudioInputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub state: String,
    pub sample_rate_hz: Option<u32>,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct AudioCaptureReport {
    pub output_path: String,
    pub sample_rate_hz: u32,
    pub frame_count: u64,
    pub duration_us: u64,
}

/// Enumerate active microphone endpoints without starting capture.
///
/// # Errors
///
/// Returns an error when the operating-system device inventory cannot be read.
#[cfg(windows)]
#[expect(
    clippy::undocumented_unsafe_blocks,
    reason = "Windows Core Audio enumeration requires small, documented FFI calls"
)]
#[expect(
    clippy::multiple_unsafe_ops_per_block,
    reason = "Core Audio endpoint metadata is exposed through several related FFI reads"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the first Windows inventory slice keeps COM setup, endpoint projection, and cleanup together"
)]
#[expect(
    clippy::items_after_statements,
    reason = "the local property-store helper stays adjacent to the Windows inventory operation"
)]
pub fn list_audio_input_devices() -> eyre::Result<Vec<AudioInputDevice>> {
    use std::ffi::c_void;
    use windows::Win32::Devices::Properties;
    use windows::Win32::Foundation::PROPERTYKEY;
    use windows::Win32::Foundation::RPC_E_CHANGED_MODE;
    use windows::Win32::Media::Audio::DEVICE_STATE_ACTIVE;
    use windows::Win32::Media::Audio::ERole;
    use windows::Win32::Media::Audio::IAudioClient;
    use windows::Win32::Media::Audio::IMMDeviceCollection;
    use windows::Win32::Media::Audio::IMMDeviceEnumerator;
    use windows::Win32::Media::Audio::MMDeviceEnumerator;
    use windows::Win32::Media::Audio::eCapture;
    use windows::Win32::System::Com::CLSCTX_ALL;
    use windows::Win32::System::Com::CLSCTX_INPROC_SERVER;
    use windows::Win32::System::Com::COINIT_MULTITHREADED;
    use windows::Win32::System::Com::CoCreateInstance;
    use windows::Win32::System::Com::CoInitializeEx;
    use windows::Win32::System::Com::CoTaskMemFree;
    use windows::Win32::System::Com::CoUninitialize;
    use windows::Win32::System::Com::STGM_READ;
    use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
    use windows::Win32::System::Variant::VT_LPWSTR;
    use windows::Win32::UI::Shell::PropertiesSystem::IPropertyStore;

    struct ComApartment {
        uninitialize_on_drop: bool,
    }

    impl ComApartment {
        fn initialize() -> eyre::Result<Self> {
            let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
            if result.is_ok() {
                return Ok(Self {
                    uninitialize_on_drop: true,
                });
            }
            if result == RPC_E_CHANGED_MODE {
                return Ok(Self {
                    uninitialize_on_drop: false,
                });
            }
            eyre::bail!("failed to initialize COM for audio device enumeration: {result:?}")
        }
    }

    impl Drop for ComApartment {
        fn drop(&mut self) {
            if self.uninitialize_on_drop {
                // Safety: this thread initialized COM successfully in `initialize`.
                unsafe { CoUninitialize() };
            }
        }
    }

    let _com = ComApartment::initialize()?;
    let enumerator: IMMDeviceEnumerator = unsafe {
        CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER)
            .map_err(|error| eyre::eyre!("failed to create Core Audio enumerator: {error}"))?
    };
    let default_id = unsafe {
        enumerator
            .GetDefaultAudioEndpoint(eCapture, ERole(1))
            .ok()
            .and_then(|device| device.GetId().ok())
            .and_then(|id| id.to_string().ok())
    };
    let collection: IMMDeviceCollection = unsafe {
        enumerator
            .EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)
            .map_err(|error| eyre::eyre!("failed to enumerate active microphones: {error}"))?
    };
    let count = unsafe { collection.GetCount()? };
    let mut devices = Vec::with_capacity(usize::try_from(count).unwrap_or_default());
    for index in 0..count {
        let device = unsafe { collection.Item(index)? };
        let id = unsafe { device.GetId()?.to_string()? };
        let properties = unsafe { device.OpenPropertyStore(STGM_READ).ok() };
        let name = properties
            .as_ref()
            .and_then(|store| {
                let key = std::ptr::from_ref(&Properties::DEVPKEY_Device_FriendlyName)
                    .cast::<PROPERTYKEY>();
                property_store_string_value(store, key).ok()
            })
            .unwrap_or_else(|| "Unknown microphone".to_string());
        let sample_rate_hz = unsafe {
            device
                .Activate::<IAudioClient>(CLSCTX_ALL, None)
                .ok()
                .and_then(|client| client.GetMixFormat().ok())
                .and_then(|format| {
                    if format.is_null() {
                        return None;
                    }
                    let sample_rate = std::ptr::addr_of!((*format).nSamplesPerSec).read_unaligned();
                    CoTaskMemFree(Some(format.cast::<c_void>()));
                    Some(sample_rate)
                })
        };
        devices.push(AudioInputDevice {
            is_default: default_id.as_deref() == Some(id.as_str()),
            id,
            name,
            state: "active".to_string(),
            sample_rate_hz,
        });
    }
    #[expect(
        clippy::multiple_unsafe_ops_per_block,
        reason = "PROPVARIANT string extraction requires a bounded union read and cleanup"
    )]
    fn property_store_string_value(
        properties: &IPropertyStore,
        key: *const PROPERTYKEY,
    ) -> eyre::Result<String> {
        let mut value = unsafe { properties.GetValue(key)? };
        let variant_type = unsafe { value.Anonymous.Anonymous.vt };
        if variant_type != VT_LPWSTR {
            unsafe { PropVariantClear(&raw mut value)? };
            eyre::bail!("Core Audio property is not a UTF-16 string")
        }
        let name = unsafe {
            let pwstr = value.Anonymous.Anonymous.Anonymous.pwszVal;
            if pwstr.is_null() {
                String::new()
            } else {
                pwstr.to_string()?
            }
        };
        unsafe { PropVariantClear(&raw mut value)? };
        Ok(name)
    }

    Ok(devices)
}

/// Enumerate active microphone endpoints on unsupported platforms.
///
/// # Errors
///
/// Always returns an unsupported-platform diagnostic outside Windows.
#[cfg(not(windows))]
pub fn list_audio_input_devices() -> eyre::Result<Vec<AudioInputDevice>> {
    eyre::bail!("microphone enumeration is currently implemented for Windows only")
}

/// Capture one explicitly bounded microphone interval to a native-rate mono-f32 WAV.
///
/// # Errors
///
/// Returns an error when the selected endpoint cannot be opened or the output cannot be written.
#[cfg(windows)]
pub fn record_audio_input(
    endpoint_id: Option<&str>,
    output_path: &Path,
    duration: Duration,
) -> eyre::Result<AudioCaptureReport> {
    windows_capture::record_audio_input(endpoint_id, output_path, duration)
}

/// Capture is not implemented outside Windows yet.
///
/// # Errors
///
/// Always returns an unsupported-platform diagnostic outside Windows.
#[cfg(not(windows))]
pub fn record_audio_input(
    _endpoint_id: Option<&str>,
    _output_path: &Path,
    _duration: Duration,
) -> eyre::Result<AudioCaptureReport> {
    eyre::bail!("microphone capture is currently implemented for Windows only")
}

#[cfg(windows)]
mod windows_capture {
    use super::AudioCaptureReport;
    use eyre::Context;
    use std::ffi::c_void;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;
    use windows::Win32::Foundation::RPC_E_CHANGED_MODE;
    use windows::Win32::Media::Audio::AUDCLNT_SHAREMODE_SHARED;
    use windows::Win32::Media::Audio::ERole;
    use windows::Win32::Media::Audio::IAudioCaptureClient;
    use windows::Win32::Media::Audio::IAudioClient;
    use windows::Win32::Media::Audio::IMMDeviceEnumerator;
    use windows::Win32::Media::Audio::MMDeviceEnumerator;
    use windows::Win32::Media::Audio::WAVEFORMATEX;
    use windows::Win32::Media::Audio::WAVEFORMATEXTENSIBLE;
    use windows::Win32::Media::Audio::eCapture;
    use windows::Win32::System::Com::CLSCTX_ALL;
    use windows::Win32::System::Com::CLSCTX_INPROC_SERVER;
    use windows::Win32::System::Com::COINIT_MULTITHREADED;
    use windows::Win32::System::Com::CoCreateInstance;
    use windows::Win32::System::Com::CoInitializeEx;
    use windows::Win32::System::Com::CoTaskMemFree;
    use windows::Win32::System::Com::CoUninitialize;
    use windows::core::GUID;
    use windows::core::PCWSTR;

    const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 0x0000_0002;
    const WASAPI_SHARED_BUFFER_100NS: i64 = 10_000_000;
    const WAVE_FORMAT_PCM: u16 = 0x0001;
    const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
    const WAVE_FORMAT_EXTENSIBLE: u16 = 0xfffe;
    const KSDATAFORMAT_SUBTYPE_PCM: GUID = GUID::from_u128(0x00000001_0000_0010_8000_00aa00389b71);
    const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: GUID =
        GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);

    struct ComApartment {
        uninitialize_on_drop: bool,
    }

    impl ComApartment {
        #[expect(
            clippy::undocumented_unsafe_blocks,
            reason = "COM apartment initialization is a process API with no borrowed pointers"
        )]
        fn initialize() -> eyre::Result<Self> {
            let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
            if result.is_ok() {
                return Ok(Self {
                    uninitialize_on_drop: true,
                });
            }
            if result == RPC_E_CHANGED_MODE {
                return Ok(Self {
                    uninitialize_on_drop: false,
                });
            }
            eyre::bail!("failed to initialize COM for microphone capture: {result:?}")
        }
    }

    impl Drop for ComApartment {
        fn drop(&mut self) {
            if self.uninitialize_on_drop {
                // Safety: this thread initialized COM successfully in `initialize`.
                unsafe { CoUninitialize() };
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum AudioSampleFormat {
        Float32,
        Pcm16,
        Pcm24,
        Pcm32,
    }

    #[derive(Clone, Copy, Debug)]
    struct AudioCaptureFormat {
        sample_rate_hz: u32,
        channels: u16,
        block_align: u16,
        sample_format: AudioSampleFormat,
    }

    struct MixFormat(*mut WAVEFORMATEX);

    impl Drop for MixFormat {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // Safety: `GetMixFormat` allocates this buffer with COM task memory.
                unsafe { CoTaskMemFree(Some(self.0.cast::<c_void>())) };
            }
        }
    }

    #[expect(
        clippy::undocumented_unsafe_blocks,
        reason = "WASAPI capture setup reads the endpoint mix format and initializes a shared client"
    )]
    pub fn record_audio_input(
        endpoint_id: Option<&str>,
        output_path: &Path,
        duration: Duration,
    ) -> eyre::Result<AudioCaptureReport> {
        if duration.is_zero() {
            eyre::bail!("capture duration must be greater than zero")
        }
        let _com = ComApartment::initialize()?;
        let enumerator: IMMDeviceEnumerator = unsafe {
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER)
                .wrap_err("failed to create Core Audio enumerator")?
        };
        let device = if let Some(endpoint_id) = endpoint_id {
            let wide: Vec<u16> = endpoint_id.encode_utf16().chain(Some(0)).collect();
            let endpoint = PCWSTR::from_raw(wide.as_ptr());
            unsafe { enumerator.GetDevice(endpoint)? }
        } else {
            unsafe {
                enumerator
                    .GetDefaultAudioEndpoint(eCapture, ERole(1))
                    .wrap_err("failed to resolve the default microphone")?
            }
        };
        let audio_client: IAudioClient = unsafe { device.Activate(CLSCTX_ALL, None)? };
        let mix_format = MixFormat(unsafe { audio_client.GetMixFormat()? });
        if mix_format.0.is_null() {
            eyre::bail!("microphone returned a null mix format")
        }
        let capture_format = unsafe { audio_capture_format(mix_format.0)? };
        let initialize_result = unsafe {
            audio_client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,
                WASAPI_SHARED_BUFFER_100NS,
                0,
                mix_format.0,
                None,
            )
        };
        initialize_result?;
        let capture_client: IAudioCaptureClient = unsafe { audio_client.GetService()? };
        unsafe { audio_client.Start()? };
        let stop_at = Instant::now() + duration;
        let capture_result = collect_samples(&capture_client, capture_format, stop_at);
        let _ = unsafe { audio_client.Stop() };
        let samples = capture_result?;
        write_capture_wav(output_path, capture_format.sample_rate_hz, &samples)?;
        let frame_count = u64::try_from(samples.len()).unwrap_or(u64::MAX);
        let duration_us = frame_count
            .saturating_mul(1_000_000)
            .checked_div(u64::from(capture_format.sample_rate_hz))
            .unwrap_or(0);
        Ok(AudioCaptureReport {
            output_path: output_path.display().to_string(),
            sample_rate_hz: capture_format.sample_rate_hz,
            frame_count,
            duration_us,
        })
    }

    #[expect(
        clippy::undocumented_unsafe_blocks,
        reason = "WASAPI packet access exposes borrowed raw buffers until ReleaseBuffer"
    )]
    fn collect_samples(
        capture_client: &IAudioCaptureClient,
        capture_format: AudioCaptureFormat,
        stop_at: Instant,
    ) -> eyre::Result<Vec<f32>> {
        let mut samples = Vec::new();
        while Instant::now() < stop_at {
            let mut packet_frames = unsafe { capture_client.GetNextPacketSize()? };
            while packet_frames > 0 {
                let mut data = std::ptr::null_mut();
                let mut frames_to_read = 0;
                let mut flags = 0;
                let buffer_result = unsafe {
                    capture_client.GetBuffer(
                        &raw mut data,
                        &raw mut frames_to_read,
                        &raw mut flags,
                        None,
                        None,
                    )
                };
                buffer_result?;
                if flags & AUDCLNT_BUFFERFLAGS_SILENT != 0 {
                    samples.extend(std::iter::repeat_n(
                        0.0,
                        usize::try_from(frames_to_read).unwrap_or_default(),
                    ));
                } else if !data.is_null() {
                    samples.extend(unsafe {
                        capture_frames_as_mono(data, frames_to_read, capture_format)
                    });
                }
                unsafe { capture_client.ReleaseBuffer(frames_to_read)? };
                packet_frames = unsafe { capture_client.GetNextPacketSize()? };
            }
            thread::sleep(Duration::from_millis(10));
        }
        Ok(samples)
    }

    #[expect(
        clippy::undocumented_unsafe_blocks,
        clippy::multiple_unsafe_ops_per_block,
        reason = "WAVEFORMATEX and WAVEFORMATEXTENSIBLE are packed Windows structs"
    )]
    unsafe fn audio_capture_format(
        format: *const WAVEFORMATEX,
    ) -> eyre::Result<AudioCaptureFormat> {
        let wave_format = unsafe { format.read_unaligned() };
        let sample_format = if wave_format.wFormatTag == WAVE_FORMAT_EXTENSIBLE {
            let extensible = format.cast::<WAVEFORMATEXTENSIBLE>();
            let sub_format =
                unsafe { std::ptr::addr_of!((*extensible).SubFormat).read_unaligned() };
            if sub_format == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT && wave_format.wBitsPerSample == 32 {
                AudioSampleFormat::Float32
            } else if sub_format == KSDATAFORMAT_SUBTYPE_PCM {
                pcm_sample_format(wave_format.wBitsPerSample)?
            } else {
                eyre::bail!("unsupported extensible microphone sample format")
            }
        } else if wave_format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT
            && wave_format.wBitsPerSample == 32
        {
            AudioSampleFormat::Float32
        } else if wave_format.wFormatTag == WAVE_FORMAT_PCM {
            pcm_sample_format(wave_format.wBitsPerSample)?
        } else {
            eyre::bail!("unsupported microphone sample format tag")
        };
        Ok(AudioCaptureFormat {
            sample_rate_hz: wave_format.nSamplesPerSec,
            channels: wave_format.nChannels.max(1),
            block_align: wave_format.nBlockAlign,
            sample_format,
        })
    }

    fn pcm_sample_format(bits_per_sample: u16) -> eyre::Result<AudioSampleFormat> {
        match bits_per_sample {
            16 => Ok(AudioSampleFormat::Pcm16),
            24 => Ok(AudioSampleFormat::Pcm24),
            32 => Ok(AudioSampleFormat::Pcm32),
            _ => eyre::bail!("unsupported microphone PCM bit depth: {bits_per_sample}"),
        }
    }

    #[expect(
        clippy::undocumented_unsafe_blocks,
        reason = "capture packets contain packed native PCM bytes that must be normalized to f32"
    )]
    unsafe fn capture_frames_as_mono(
        data: *const u8,
        frame_count: u32,
        capture_format: AudioCaptureFormat,
    ) -> Vec<f32> {
        let frame_count = usize::try_from(frame_count).unwrap_or_default();
        let channels = usize::from(capture_format.channels);
        let block_align = usize::from(capture_format.block_align);
        let bytes_per_sample = block_align / channels.max(1);
        let mut samples = Vec::with_capacity(frame_count);
        for frame_index in 0..frame_count {
            let frame_base = unsafe { data.add(frame_index * block_align) };
            let mut sum = 0.0;
            for channel_index in 0..channels {
                let sample_base = unsafe { frame_base.add(channel_index * bytes_per_sample) };
                sum += unsafe { read_capture_sample(sample_base, capture_format.sample_format) };
            }
            samples.push(sum / f32::from(capture_format.channels));
        }
        samples
    }

    #[expect(
        clippy::undocumented_unsafe_blocks,
        clippy::multiple_unsafe_ops_per_block,
        clippy::cast_lossless,
        clippy::cast_precision_loss,
        reason = "sample decoding reads packed PCM bytes and normalizes them to f32"
    )]
    unsafe fn read_capture_sample(sample_base: *const u8, sample_format: AudioSampleFormat) -> f32 {
        match sample_format {
            AudioSampleFormat::Float32 => {
                unsafe { sample_base.cast::<f32>().read_unaligned() }.clamp(-1.0, 1.0)
            }
            AudioSampleFormat::Pcm16 => {
                f32::from(unsafe { sample_base.cast::<i16>().read_unaligned() }) / 32768.0
            }
            AudioSampleFormat::Pcm24 => {
                let byte0 = unsafe { *sample_base.add(0) } as i32;
                let byte1 = unsafe { *sample_base.add(1) } as i32;
                let byte2 = unsafe { *sample_base.add(2) } as i32;
                let value = (byte0 | (byte1 << 8) | (byte2 << 16)) << 8 >> 8;
                value as f32 / 8_388_608.0
            }
            AudioSampleFormat::Pcm32 => {
                (unsafe { sample_base.cast::<i32>().read_unaligned() }) as f32 / 2_147_483_648.0
            }
        }
    }

    fn write_capture_wav(path: &Path, sample_rate_hz: u32, samples: &[f32]) -> eyre::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for sample in samples.iter().copied() {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

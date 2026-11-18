# teamy-transcriber

Local microphone transcription with Whisper words and PhoneticXeus IPA phonemes.

## Install

On Windows, you'll need PowerShell 7, an NVIDIA GPU, the CUDA Toolkit, Rust with the Visual Studio C++ Build Tools, and the Hugging Face CLI (`hf`).

Run these commands from this repository. The installer places the executable in Cargo's bin directory, which must be on `PATH`.

~~~powershell
./update.ps1 # Build and install teamy-transcriber and stage the CUDA DLLs beside it.
hf download TeamDman/teamy-transcriber-whisper-large-v3 --revision ca74fe211e39bcb869ae19204c39b7b4a2967107
teamy-transcriber model prepare
hf download changelinglab/PhoneticXeus model.safetensors config.json ipa_vocab.json --revision 3a8d860fa68f8936ceb4196651221215bab9dae4 --local-dir ./phone-model
teamy-transcriber model prepare-phones --source-dir ./phone-model
~~~

## Commands and hotkeys

~~~powershell
❯ teamy-transcriber help list --short
teamy-transcriber.exe cache clean
teamy-transcriber.exe cache open
teamy-transcriber.exe cache show
teamy-transcriber.exe doctor
teamy-transcriber.exe home open
teamy-transcriber.exe home show
teamy-transcriber.exe gui # ⭐
teamy-transcriber.exe model show
teamy-transcriber.exe model prepare
teamy-transcriber.exe model prepare-vad
teamy-transcriber.exe model prepare-phones
teamy-transcriber.exe microphone list
teamy-transcriber.exe microphone record
teamy-transcriber.exe microphone transcribe # ⭐
teamy-transcriber.exe recording clip add
teamy-transcriber.exe recording clean
teamy-transcriber.exe recording create
teamy-transcriber.exe recording export
teamy-transcriber.exe recording list
teamy-transcriber.exe recording prepare
teamy-transcriber.exe recording show
teamy-transcriber.exe recording transcribe
teamy-transcriber.exe transcribe
teamy-transcriber.exe verify speech vctk-p230-385
teamy-transcriber.exe phones
~~~
~~~hs
~
❯ teamy-transcriber.exe gui --help
teamy-transcriber.exe gui

Open the native Teamy-Transcriber desktop window.

USAGE:
    teamy-transcriber.exe gui


Implementation:
    src\cli\gui.rs
    https://github.com/TeamDman/teamy-transcriber/blob/7449bd1/src/cli/gui.rs
~~~

~~~hs
~
❯ teamy-transcriber.exe transcribe microphone --help
teamy-transcriber.exe transcribe

Transcribe a media file, printing text and cleaning up successful work.

USAGE:
    teamy-transcriber.exe transcribe [OPTIONS] [SOURCE]

ARGUMENTS:
        <SOURCE>
            Audio/video file. Omit when using --resume.

OPTIONS:
        --resume <STRING>
            Resume a retained recording UUID instead of creating another
            recording.
        --kind <audio,video,microphone>
            Override the source kind inferred from its extension.
        --model-dir <STRING>
            Override the selected Whisper model; text mode only.
        --[no-]phonemes
            Emit `PhoneticXeus` IPA phones instead of Whisper text.
            aliases: phones
        --phone-model-dir <STRING>
            Phone model folder; only applies with --phonemes.
        --output <STRING>
            Also write the output to a new file; existing files are never
            overwritten.
        --[no-]keep-recording
            Keep the recording and intermediate audio after success.
        --[no-]no-vad
            Transcribe every 30-second window, bypassing speech detection
            (useful for singing/music).
        --max-decode-tokens <USIZE>
            Maximum generated tokens per clip (default 448).


Implementation:
    src\cli\transcribe\transcribe_cli.rs
    https://github.com/TeamDman/teamy-transcriber/blob/7449bd1/src/cli/transcribe/transcribe_cli.rs

~~~
Use `--help` after a subcommand for its options. To select a microphone, pass `--device-id` with an ID from `microphone list`.

While the Windows GUI is running, `Ctrl+Shift+Space` starts or stops microphone capture. The shortcut is enabled by default. Toggle it with `HOTKEY ON/OFF` in the window or from the tray menu.

## PowerShell example

Add this function to `$PROFILE`, then open a new PowerShell window and run `td`. It displays and saves a timestamped transcript in Downloads. Each audio window emits IPA phonemes on `1|` first, then words on `2|`. After a chunk finishes, 10 seconds with no further transcription output triggers one `...` line. Add `--inactivity-ms 5000` in any mode for a five-second gap. Words-only and phonemes-only modes show the marker when you supply this option. Press `Ctrl+C` to stop recording and finish processing queued audio.

~~~powershell
function td {
    $start = Get-Date
    $downloads = Join-Path $HOME 'Downloads'
    $file = Join-Path $downloads "transcription_$($start.ToString('yyyyMMdd_HHmmss')).txt"

    Set-Content -LiteralPath $file -Value "Start: $($start.ToString('o'))"
    Write-Host "Transcription saving to `"$file`""
    try {
        teamy-transcriber microphone transcribe words-and-phonemes | Tee-Object -FilePath $file -Append | Out-Host
    }
    finally {
        Add-Content -LiteralPath $file -Value "End: $((Get-Date).ToString('o'))"
    }
}
~~~


~~~r
❯ td
Transcription saving to "C:\Users\Teamy\Downloads\transcription_20261117_193549.txt"
Transcribing at speech pauses (320 ms quiet), with a 5000 ms maximum window.
Completed microphone recordings are removed after output; failed or unfinished recordings are retained for recovery.
Recording microphone. Ctrl+C stops capture and drains transcription.
1| tʰʃɛkwəntʰu
2| Check one, two.
1| hloʊðɛɹ
2| Hello there.
1| aɪlaɪkðætsnɪpɪtðætwihævənmaɪpʰaʊɜ˞ʃɛl̴pʰɹoʊfaɪl̴aɪθɪ̃ŋkwiʃʊdɹiɹaɪtðəɹɛdmifɹəmðəkɹaʊ̃ndəptʰu
1| vi
2| I like that snippet that we have in my PowerShell profile. I think we should rewrite the read me from the ground up
1| ðəpɛɹmɪ̃nəməmɪ̃nstəleɪʃənɪ̃nstɹʌkʃənzpʰlʌs
2| bare minimum installation instructions plus
1| ðɪspʰaʊɜ˞ʃɛl̴snɪpɪtæzæ̃nɪɡzæ̃mpəl̴
2| this PowerShell snippet as an example.
~~~


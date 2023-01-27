from TTS.api import TTS
import os

if __name__ == '__main__':
    os.makedirs("audios", exist_ok=True)
    model_path = input("\nPath to the model checkpoint .pth\n")
    config_path = input("\nPath to config.json\n")
    name = input("\nReference audio .wav or Enter\n")
    lang = input("\nLanguage: ['en','qu']\n")
    model_name = model_path.split('\\')[-1].split('.')[0]

    if lang =="qu":
        text = "Qantapuni qhawashayki warmiypaq."
        if name:
            command=f"tts --text \"{text}\" --model_path {model_path} --config_path {config_path} --out_path audios\{name}_{lang}_test_{model_name}.wav --speaker_wav C:\\Users\\Aorus\\Documents\\Voice_Cloning_Quechua\\{name}.wav --language_idx {lang}"
            os.system("cmd /c "+command+"")
        name = "gui"
        command=f"tts --text \"{text}\" --model_path {model_path} --config_path {config_path} --out_path audios\{name}_{lang}_test_{model_name}.wav --speaker_id {name} --language_idx {lang}"
        os.system("cmd /c "+command+"")
        name = "LTTS_100"
        command=f"tts --text \"{text}\" --model_path {model_path} --config_path {config_path} --out_path audios\{name}_{lang}_test_{model_name}.wav --speaker_id {name} --language_idx {lang}"
        os.system("cmd /c "+command+"")
    else:
        text = "This is a sentence meant for testing, whether the new vocoder works the way I hope it does."
        if name:
            command=f"tts --text \"{text}\" --model_path {model_path} --config_path {config_path} --out_path audios\{name}_{lang}_test_{model_name}.wav --speaker_wav C:\\Users\\Aorus\\Documents\\Voice_Cloning_Quechua\\{name}.wav --language_idx {lang}"
            os.system("cmd /c "+command+"")
        name = "gui"
        command=f"tts --text \"{text}\" --model_path {model_path} --config_path {config_path} --out_path audios\{name}_{lang}_test_{model_name}.wav --speaker_id {name} --language_idx {lang}"
        os.system("cmd /c "+command+"")
        name = "LTTS_100"
        command=f"tts --text \"{text}\" --model_path {model_path} --config_path {config_path} --out_path audios\{name}_{lang}_test_{model_name}.wav --speaker_id {name} --language_idx {lang}"
        os.system("cmd /c "+command+"")

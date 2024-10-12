from academicodec.models.hificodec.models import Encoder
from academicodec.models.hificodec_lfq.lookup_free_quantize import LFQ
from academicodec.models.hificodec.models import Generator
import torch
import json
from academicodec.models.hificodec.env import AttrDict, build_env
import tqdm, os, librosa, soundfile as sf, torchaudio

from academicodec.utils import load_checkpoint
seed = 42

torch.cuda.manual_seed(42)
device = torch.device('cuda')

config_path = "/mnt/bear2/users/cjs/model/AcademiCodec/egs/HiFi-Codec-LFQ-16k-320d/commit1.0_entropy0.4/config.json"
cp_g = "/mnt/bear2/users/cjs/model/AcademiCodec/egs/HiFi-Codec-LFQ-16k-320d/commit1.0_entropy0.4/g_00195000"

with open(config_path) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

encoder = Encoder(h).to(device)
generator = Generator(h).to(device)
# quantizer = Quantizer(h).to(device)
quantizer = LFQ(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)

generator.load_state_dict(state_dict_g['generator'])
encoder.load_state_dict(state_dict_g['encoder'])
quantizer.load_state_dict(state_dict_g['quantizer'])

generator.remove_weight_norm()
encoder.remove_weight_norm()

generator = generator.cuda().eval()
encoder = encoder.cuda().eval()
quantizer = quantizer.cuda().eval()

with open ("/mnt/lynx4/datasets/SpeechLLM/LibriTTS_16khz/labels/test-clean.lst") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

root_path = "/mnt/lynx4/datasets/SpeechLLM/LibriTTS/raw"
save_root_path = "/mnt/bear2/users/cjs/model/WavTokenizer/sample/reconstructed_lfq"
code_save_root_path = "/mnt/bear2/users/cjs/model/WavTokenizer/sample/code_lfq"

for l in tqdm.tqdm(lines):
    audio_path = os.path.join(root_path, l)

    wav, sr = librosa.load(audio_path, sr=16000)
    wav = torch.tensor(wav).unsqueeze(0)
    wav = wav.cuda()

    c = encoder(wav.unsqueeze(1))
    # print("c.shape: ", c.shape)
    q, loss_q, c = quantizer(c)

    # print("q.shape: ", q.shape)
    y_g_hat = generator(q)

    code_save_path = os.path.join(code_save_root_path, l[:-4]+".pth")
    os.makedirs(os.path.dirname(code_save_path), exist_ok=True)
    torch.save(c.unsqueeze(1).cpu(), code_save_path)
    
    save_path = os.path.join(save_root_path, l)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torchaudio.save(save_path, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
    sf.write(save_path, y_g_hat.squeeze().cpu().detach().numpy(), 16000)

import torch
# from inference.tts.fs import FastSpeechInfer
# from modules.tts.fs2_orig import FastSpeech2Orig
from inference.svs.base_svs_infer import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from usr.diff.shallow_diffusion_tts import GaussianDiffusion
from usr.diffsinger_task import DIFF_DECODERS
from modules.fastspeech.pe import PitchExtractor
import utils
from modules.fastspeech.tts_modules import LengthRegulator
import librosa
import numpy as np


class DiffSingerE2EInfer(BaseSVSInfer):
    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        return model

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp['ph_seq']
        note_lst = inp['note_seq'].split()
        midi_dur_lst = inp['note_dur_seq'].split()
        is_slur = np.array(inp['is_slur_seq'].split(),'float')
        ph_dur=np.array(inp['ph_dur'].split(),'float')
        print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst),len(ph_dur))
        if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst)==len(ph_dur):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur
        
    def preprocess_input(self, inp, input_type='word'):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', 'opencpop')

        # single spk
        spk_id = self.spk_map[spk_name]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ret = self.preprocess_word_level_input(inp)
        elif input_type == 'phoneme':  # like transcriptions.txt in Opencpop dataset.
            ret = self.preprocess_phoneme_level_input(inp)
        else:
            print('Invalid input type.')
            return None

        if ret:
            if input_type == 'word':
                ph_seq, note_lst, midi_dur_lst, is_slur = ret
            else:
                ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur = ret
        else:
            print('==========> Preprocess_word_level or phone_level input wrong.')
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                     for x in note_lst]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {'item_name': item_name, 'text': inp['text'], 'ph': ph_seq, 'spk_id': spk_id,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), 'ph_dur': None}
        item['ph_len'] = len(item['ph_token'])
        if input_type == 'phoneme':
            item['ph_dur'] = np.asarray(ph_dur)
        return item
    
    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :hparams['max_frames']].to(self.device)
        midi_dur = torch.FloatTensor(item['midi_dur'])[None, :hparams['max_frames']].to(self.device)
        is_slur = torch.LongTensor(item['is_slur'])[None, :hparams['max_frames']].to(self.device)
        mel2ph = None
        if item['ph_dur'] is not None:
            ph_acc=np.around(np.add.accumulate(24000*item['ph_dur']/128)).astype('int')
            ph_dur=np.diff(ph_acc,prepend=0)
            ph_dur = torch.LongTensor(ph_dur)[None, :hparams['max_frames']].to(self.device)
            lr=LengthRegulator()
            mel2ph=lr(ph_dur,txt_tokens==0).detach()

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur,
            'mel2ph': mel2ph
        }
        return batch
        
    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        with torch.no_grad():
            output = self.model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'],mel2ph=sample['mel2ph'])
            mel_out = output['mel_out']  # [B, T,80]
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                f0_pred = self.pe(mel_out)['f0_denorm_pred']  # pe predict from Pred mel
            else:
                f0_pred = output['f0_denorm']
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]

if __name__ == '__main__':
    inp = {
        'text': '小酒窝长睫毛AP是你最美的记号',
        'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
        'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
        'input_type': 'word'
    }  # user input: Chinese characters
    c = {
        'text': 'SP 还 记 得 那 场 音 乐 会 的 烟 火  SP 还 记 得 那 个 凉 凉 的 深 秋  SP 还 记 得 人 潮 把 你 推 向 了 我  SP 游 乐 园 拥 挤 的 正 是 时 候  SP 一 个 夜 晚 坚 持 不 睡 的 等 候  SP 一 起 泡 温 泉 奢 侈 的 享 受  SP 有 一 次 日 记 里 愚 蠢 的 困 惑  SP 因 为 你 的 微 笑 幻 化 成 风  SP 你 大 大 的 勇 敢 保 护 着 我 SP 我 小 小 的 关 怀 喋 喋 不 休 SP 感 谢 我 们 一 起 走 了 那 么 久 SP 又 再 一 次 回 到 凉 凉 深 秋 SP 给 你 我 的 手 SP 像 温 柔 野 兽 AP 把 自 由 交 给 草 原 的 辽 阔  SP 我 们 小 手 拉 大 手 SP 一 起 郊 游 SP 今 天 别 想 太 多 SP 你 是 我 的 梦 SP 像 北 方 的 风 SP 吹 着 南 方 暖 洋 洋 的 哀 愁   SP 我 们 小 手 拉 大 手 SP 今 天 加 油 SP 向 昨 天 挥 挥 手  ',
        'ph_seq': 'SP h ai j i d e n a ch ang y in y ve h ui d e y an h uo uo SP h ai j i d e n a g e l iang l iang d e sh en q iu iu SP h ai j i d e r en ch ao b a n i t ui x iang l e w o o SP y ou l e y van y ong j i d e zh eng sh i sh i h ou ou SP y i g e y e w an j ian ch i b u sh ui d e d eng h ou ou SP y i q i p ao w en q van sh e ch i d e x iang sh ou ou SP y ou y i c i r i j i l i y v ch un d e k un h uo uo SP y in w ei n i d e w ei x iao h uan h ua ch eng f eng eng SP n i d a d a d e y ong g an b ao h u zh uo w o SP w o x iao x iao d e g uan h uai d ie d ie b u x iu SP g an x ie w o m en y i q i z ou l e n a m e j iu SP y ou z ai y i c i h ui d ao l iang l iang sh en q iu SP g ei n i w o d e sh ou SP x iang w en r ou y e sh ou AP b a z i y ou j iao g ei c ao y van d e l iao k uo uo SP w o m en x iao sh ou l a d a sh ou SP y i q i j iao y ou SP j in t ian b ie x iang t ai d uo SP n i sh i w o d e m eng SP x iang b ei f ang d e f eng SP ch ui zh uo n an f ang n uan y ang y ang d e ai ch ou ou ou SP w o m en x iao sh ou l a d a sh ou SP j in t ian j ia y ou SP x iang z uo t ian h ui h ui sh ou ou ou',
        'note_seq': 'rest G3 G3 G3 G3 A3 A3 C4 C4 D4 D4 E4 E4 A4 A4 G4 G4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 E4 E4 E4 E4 D4 rest D4 D4 E4 E4 D4 D4 C#4 C#4 D4 D4 G4 G4 B3 B3 D4 D4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 A3 A3 A3 A3 G3 rest G3 G3 G3 G3 A3 A3 C4 C4 D4 D4 E4 E4 A4 A4 G4 G4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 E4 E4 E4 E4 D4 rest D4 D4 E4 E4 D4 D4 C#4 C#4 D4 D4 G4 G4 B3 B3 D4 D4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 D4 D4 D4 D4 C4 rest E4 E4 F4 F4 E4 E4 D4 D4 E4 E4 F4 F4 E4 E4 D4 D4 E4 E4 F4 F4 rest F4 F4 G4 G4 F4 F4 G4 G4 F4 F4 E4 E4 D4 D4 C4 C4 D4 D4 E4 E4 rest E4 E4 E4 E4 D4 D4 C#4 C#4 E4 E4 E4 E4 D4 D4 D4 D4 D4 D4 C#4 C#4 D4 D4 rest D4 D4 D4 D4 E4 E4 F4 F4 D4 D4 A4 A4 G4 G4 G4 G4 F#4 F#4 G4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest G4 G4 A4 A4 G4 G4 A4 A4 B4 B4 C5 C5 E4 E4 E4 E4 A4 A4 A4 A4 G4 rest C4 C4 D4 D4 C4 C4 F4 F4 E4 E4 D4 D4 C4 C4 rest E4 E4 E4 E4 D4 D4 C4 C4 rest C4 C4 D4 D4 A3 A3 C4 C4 E4 E4 G4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest G4 G4 A4 A4 G4 G4 A4 A4 B4 B4 C5 C5 E4 E4 E4 E4 A4 A4 A4 G4 G4 rest C4 C4 D4 D4 C4 C4 F4 F4 E4 E4 D4 D4 C4 C4 rest E4 E4 E4 E4 D4 D4 C4 C4 rest C4 C4 D4 D4 A3 A3 C4 C4 D4 D4 D4 D4 C4 C4',
        'note_dur_seq': '7.911923 0.320769 0.320769 0.260769 0.260769 0.200769 0.200769 0.230769 0.230769 0.260769 0.260769 0.461538 0.461538 0.200768 0.200768 0.491539 0.491539 0.230769 0.230769 0.200768 0.200768 0.30577 0.30577 0.288462 0.328846 0.23077 0.23077 0.260768 0.260768 0.20077 0.20077 0.245769 0.245769 0.230768 0.230768 0.401539 0.401539 0.305769 0.305769 0.356539 0.356539 0.476538 0.476538 0.596538 0.596538 0.230769 0.17077 0.215767 0.215767 0.26077 0.26077 0.230769 0.230769 0.200768 0.200768 0.26077 0.26077 0.416538 0.416538 0.245768 0.245768 0.38654 0.38654 0.305768 0.305768 0.26077 0.26077 0.506538 0.506538 0.230769 0.185769 0.21577 0.21577 0.245768 0.245768 0.230769 0.230769 0.200768 0.200768 0.26077 0.26077 0.401537 0.401537 0.260769 0.260769 0.386538 0.386538 0.506541 0.506541 0.566538 0.566538 0.230769 0.185769 0.215768 0.215768 0.24577 0.24577 0.230769 0.230769 0.230771 0.230771 0.200766 0.200766 0.476538 0.476538 0.215771 0.215771 0.491537 0.491537 0.230769 0.230769 0.20077 0.20077 0.536537 0.536537 0.230769 0.185769 0.20077 0.20077 0.230769 0.230769 0.260768 0.260768 0.20077 0.20077 0.230767 0.230767 0.38654 0.38654 0.335769 0.335769 0.356539 0.356539 0.461537 0.461537 0.61154 0.61154 0.230769 0.185771 0.230765 0.230765 0.20077 0.20077 0.260772 0.260772 0.200766 0.200766 0.230773 0.230773 0.491537 0.491537 0.200766 0.200766 0.491541 0.491541 0.200766 0.200766 0.230769 0.230769 0.536539 0.536539 0.230769 0.185767 0.230773 0.230773 0.215766 0.215766 0.245772 0.245772 0.230765 0.230765 0.20077 0.20077 0.43154 0.43154 0.260768 0.260768 0.386538 0.386538 0.491542 0.491542 0.581537 0.581537 0.346154 0.070386 0.230765 0.230765 0.230773 0.230773 0.230765 0.230765 0.230773 0.230773 0.230769 0.230769 0.461538 0.461538 0.200766 0.200766 0.431536 0.431536 0.521543 0.521543 0.679613 0.679613 0.243463 0.200766 0.200766 0.230769 0.230769 0.260768 0.260768 0.215774 0.215774 0.215766 0.215766 0.491537 0.491537 0.21577 0.21577 0.446539 0.446539 0.386538 0.386538 0.784617 0.784617 0.22846 0.215769 0.215769 0.260768 0.260768 0.20077 0.20077 0.260772 0.260772 0.200766 0.200766 0.446543 0.446543 0.260764 0.260764 0.41654 0.41654 0.260768 0.260768 0.230769 0.230769 0.695193 0.695193 0.257883 0.21577 0.21577 0.245768 0.245768 0.20077 0.20077 0.230773 0.230773 0.260764 0.260764 0.446539 0.446539 0.230773 0.230773 0.371539 0.371539 0.461539 0.461539 0.726921 0.726921 0.286152 0.245772 0.245772 0.230765 0.230765 0.230773 0.230773 0.200766 0.200766 0.536539 0.536539 0.32654 0.320765 0.320765 0.230773 0.230773 0.230765 0.230765 0.20077 0.20077 0.421155 0.421155 0.531925 0.200766 0.200766 0.260768 0.260768 0.200774 0.200774 0.260768 0.260768 0.20077 0.20077 0.491537 0.491537 0.230765 0.230765 0.43154 0.43154 0.401537 0.401537 0.365771 0.365771 0.836538 0.27231 0.215766 0.215766 0.215773 0.215773 0.230765 0.230765 0.461539 0.461539 0.245773 0.245773 0.371535 0.371535 0.525002 0.525002 0.272302 0.200778 0.200778 0.43154 0.43154 0.521536 0.521536 0.275767 0.275767 0.140773 0.245761 0.245761 0.491545 0.491545 0.20077 0.20077 0.401533 0.401533 0.536543 0.536543 0.521536 0.521536 0.41654 0.200763 0.200763 0.260776 0.260776 0.230769 0.230769 0.20077 0.20077 0.536535 0.536535 0.326544 0.320766 0.320766 0.200763 0.200763 0.260776 0.260776 0.20077 0.20077 0.421151 0.421151 0.471927 0.275768 0.275768 0.215763 0.215763 0.23077 0.23077 0.245777 0.245777 0.245769 0.245769 0.461538 0.461538 0.230769 0.230769 0.446531 0.446531 0.386549 0.365764 0.365764 0.461538 0.230769 0.416533 0.21577 0.21577 0.215777 0.215777 0.230769 0.230769 0.47653 0.47653 0.230777 0.230777 0.371543 0.371543 0.539417 0.539417 0.197889 0.260768 0.260768 0.446531 0.446531 0.506544 0.506544 0.218075 0.218075 0.153459 0.290767 0.290767 0.416548 0.416548 0.275767 0.275767 0.401533 0.401533 0.446547 0.446547 0.380763 0.380763 0.461538 0.346154',
        'is_slur_seq': '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1',
        'ph_dur': '7.911923 0.165 0.155769 0.075 0.185769 0.045 0.155769 0.075 0.155769 0.075 0.185769 0.045 0.416538 0.045 0.155768 0.075001 0.416538 0.045 0.185769 0.045 0.155768 0.075001 0.230769 0.288462 0.328846 0.075 0.15577 0.074999 0.185769 0.045 0.15577 0.074999 0.17077 0.059999 0.170769 0.06 0.341539 0.12 0.185769 0.045 0.311539 0.15 0.326538 0.135 0.461538 0.230769 0.17077 0.059999 0.155768 0.075001 0.185769 0.045 0.185769 0.045 0.155768 0.075001 0.185769 0.045 0.371538 0.09 0.155768 0.075001 0.311539 0.15 0.155768 0.075001 0.185769 0.045 0.461538 0.230769 0.185769 0.045 0.17077 0.059999 0.185769 0.045 0.185769 0.045 0.155768 0.075001 0.185769 0.045 0.356537 0.105001 0.155768 0.075001 0.311537 0.150002 0.356539 0.105 0.461538 0.230769 0.185769 0.045 0.170768 0.060001 0.185769 0.045 0.185769 0.045 0.185771 0.044998 0.155768 0.075001 0.401537 0.060001 0.15577 0.074999 0.416538 0.045 0.185769 0.045 0.15577 0.074999 0.461538 0.230769 0.185769 0.045 0.15577 0.074999 0.15577 0.074999 0.185769 0.045 0.15577 0.074999 0.155768 0.075001 0.311539 0.15 0.185769 0.045 0.311539 0.15 0.311537 0.150002 0.461538 0.230769 0.185771 0.044998 0.185767 0.045002 0.155768 0.075001 0.185771 0.044998 0.155768 0.075001 0.155772 0.074997 0.41654 0.044998 0.155768 0.075001 0.41654 0.044998 0.155768 0.075001 0.155768 0.075001 0.461538 0.230769 0.185767 0.045002 0.185771 0.044998 0.170768 0.060001 0.185771 0.044998 0.185767 0.045002 0.155768 0.075001 0.356539 0.105 0.155768 0.075001 0.311537 0.150002 0.34154 0.119999 0.461538 0.346154 0.070386 0.044998 0.185767 0.045002 0.185771 0.044998 0.185767 0.045002 0.185771 0.044998 0.185771 0.044998 0.41654 0.044998 0.155768 0.075001 0.356535 0.105003 0.41654 0.044998 0.634615 0.243463 0.044998 0.155768 0.075001 0.155768 0.075001 0.185767 0.045002 0.170772 0.059998 0.155768 0.075001 0.416536 0.045002 0.170768 0.060001 0.386538 0.075001 0.311537 0.150002 0.634615 0.22846 0.060001 0.155768 0.075001 0.185767 0.045002 0.155768 0.075001 0.185771 0.044998 0.155768 0.075001 0.371542 0.089996 0.170768 0.060001 0.356539 0.105 0.155768 0.075001 0.155768 0.075001 0.620192 0.257883 0.045002 0.170768 0.060001 0.185767 0.045002 0.155768 0.075001 0.155772 0.074997 0.185767 0.045002 0.401537 0.060001 0.170772 0.059998 0.311541 0.149998 0.311541 0.149998 0.576923 0.286152 0.060001 0.185771 0.044998 0.185767 0.045002 0.185771 0.044998 0.155768 0.075001 0.461538 0.32654 0.134998 0.185767 0.045002 0.185771 0.044998 0.185767 0.045002 0.155768 0.075001 0.346154 0.531925 0.044998 0.155768 0.075001 0.185767 0.045002 0.155772 0.074997 0.185771 0.044998 0.155772 0.074997 0.41654 0.044998 0.185767 0.045002 0.386538 0.075001 0.326536 0.135002 0.230769 0.836538 0.27231 0.044998 0.170768 0.060001 0.155772 0.074997 0.155768 0.075001 0.386538 0.075001 0.170772 0.059998 0.311537 0.150002 0.375 0.272302 0.045006 0.155772 0.074997 0.356543 0.104996 0.41654 0.044998 0.230769 0.140773 0.089996 0.155765 0.075005 0.41654 0.044998 0.155772 0.074997 0.326536 0.135002 0.401541 0.059998 0.461538 0.41654 0.044998 0.155765 0.075005 0.185771 0.044998 0.185771 0.044998 0.155772 0.074997 0.461538 0.326544 0.134995 0.185771 0.044998 0.155765 0.075005 0.185771 0.044998 0.155772 0.074997 0.346154 0.471927 0.104996 0.170772 0.059998 0.155765 0.075005 0.155765 0.075005 0.170772 0.059998 0.185771 0.044998 0.41654 0.044998 0.185771 0.044998 0.401533 0.386549 0.134995 0.230769 0.461538 0.230769 0.416533 0.045006 0.170764 0.060005 0.155772 0.074997 0.155772 0.074997 0.401533 0.060005 0.170772 0.059998 0.311545 0.149994 0.389423 0.197889 0.104996 0.155772 0.074997 0.371534 0.090004 0.41654 0.044998 0.173077 0.153459 0.135002 0.155765 0.075005 0.341543 0.119995 0.155772 0.074997 0.326536 0.135002 0.311545 0.149994 0.230769 0.461538 0.346154',
        'input_type': 'phoneme'
    }  # input like Opencpop dataset.
    DiffSingerE2EInfer.example_run(c)


# python inference/svs/ds_e2e.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name 0228_opencpop_ds100_rel
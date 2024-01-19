# AudioLDM 2: Web UI

Simple web UI for AudioLDM 2, built with [Streamlit](https://streamlit.io/).

## How to install

```bash
git clone git@github.com:Danand/audio-ldm-webui.git
cd audio-ldm-webui
python3 -m venv .venv
source .venv/bin/activate
pip install --pre -r requirements.txt
```

## How to launch

```bash
streamlit run webui.py
```

## Credits

- [**@cvssp**](https://huggingface.co/cvssp) – [AudioLDM 2 model](https://huggingface.co/cvssp/audioldm2).

## Citation

```bibtex
@article{liu2023audioldm2,
  title={"AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining"},
  author={Haohe Liu and Qiao Tian and Yi Yuan and Xubo Liu and Xinhao Mei and Qiuqiang Kong and Yuping Wang and Wenwu Wang and Yuxuan Wang and Mark D. Plumbley},
  journal={arXiv preprint arXiv:2308.05734},
  year={2023}
}
```

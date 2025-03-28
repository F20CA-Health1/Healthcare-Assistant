# Healthcare-Assistant

## Role Allocation
- Zhaorui Jiang - Project Coordinator
- Rohan Veit - Deputy Coordinator
- Kareem Al-Hasan - RAG Manager
- Jiahui Kang - NLU language understanding
- Ameen - App/GUI developement + TTS Engineer
- Zhaorui Jiang + Zewei Yan - Model Evaluation/Research/Implementation (Safety + Performance + Bias (Gender/Race) + Interpret + Finetuning + Prompting)
- Yujing Ju - TTS Engineer
- Lanting Huang - System Engineer

## requirement
pip install numpy
pip install scipy
pip install speechrecognition
pip install sounddevice
pip install panel
pip install setuptools-rust
pip install openai-whisper 
pip install pygame
pip install soundfile
pip install openai
pip install groq
pip install chromadb
pip install torch>=2.6.0
pip install transformers>=4.48.3
pip install openai>=1.61.1

## AzureOpenAI

Remember to replace the AzureOpenAI information in "NLU_module\source\model_definition.py" with your own.

```python
os.environ['OPENAI_GPT_KEY'] = 'XXXXXXXXXXXXXX' # Change to your AzureOpenAI key
os.environ['AZURE_ENDPOINT_GPT'] = 'https://XXXXXXXXXXXXXX.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview' # Change to your endpoint
```


### how to run
panel serve GUI/healthApp.py --static-dirs assets=./assets

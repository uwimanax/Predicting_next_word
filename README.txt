GROUP 1 STREAMLIT APP — QUICK START (Windows)

1. Install Python 3.11 or 3.12 (recommended; avoid Python 3.14 for this project).
2. Extract this folder somewhere easy to find, for example:
   C:\Users\YourName\Desktop\group1_streamlit_app

3. Open PowerShell in that folder:
   - Open the folder in File Explorer.
   - Click the address bar, type powershell, and press Enter.

4. Create a virtual environment:
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

5. Start the app:
   .\.venv\Scripts\python.exe -m streamlit run app.py

6. Streamlit should open a browser page, usually http://localhost:8501.

NOTES
- The first model run downloads model files from Hugging Face and needs an internet connection.
- Choose GPT-2 small first to test the app. It is much lighter than Qwen3.
- Qwen3-4B is a large model. A compatible NVIDIA GPU with enough VRAM is recommended. On CPU it may be extremely slow or run out of RAM.
- This app does not require the notebook to be running at the same time.
- The notebook's original Sunflower-9B 4-bit model is not included in this first app version.
- If Windows blocks the streamlit executable, run it using the exact Python command above rather than `streamlit run app.py`.

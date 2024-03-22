import os
import gradio as gr
from app.app import App
from utils.config_handler import load_config
from utils.enums import DirName, FileName
from utils.misc import get_dir_path

config_dir = get_dir_path(os.path.join(os.pardir, DirName.CONFIGS.value))
config = load_config(os.path.join(config_dir, FileName.CONFIG.value))

app = App(
    region_name=config.region_name,
    profile_name=config.profile_name,
    proj_name=config.proj_name,
)

demo = gr.Interface(
    fn=app.invoke,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Textbox(label="Enter Prompt"),
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="🧞‍♂️Persona Genie",
    description="Create your own AI avatar photo in 10 seconds! Begin your prompt with 'a photo of a man (or woman)...'. For example, 'a photo of a man portrayed as Iron Man'.",
)

demo.launch(share=True)

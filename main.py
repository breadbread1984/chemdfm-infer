#!/usr/bin/python3

from os import environ
from absl import flags, app
from os.path import exists
from langchain.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from models import ChemDFM

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_string('host', default = '0.0.0.0', help = 'host address')
  flags.DEFINE_integer('port', default = 8880, help = 'port number')

class Warper(object):
  def __init__(self):
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    #llm = HuggingFaceEndpoint(endpoint_url = "https://api-inference.huggingface.co/models/OpenDFM/ChemDFM-13B-v1.0", token = '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', task = 'text-generation')
    llm = ChemDFM(FLAGS.device)
    self.chain = PromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
  def query(self, question, history):
    try:
      answer = self.chain.invoke({"prompt": question})
      history.append((question, answer))
      return "", history
    except Exception as e:
      return e, history

def main(unused_argv):
  warper = Warper()
  block = gr.Blocks()
  with block as demo:
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>文献问答系统</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(height = 450, show_copy_button = True)
        msg = gr.Textbox(label = "需要问什么？")
        with gr.Row():
          submit_btn = gr.Button("发送")
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot], value = "清空问题")
      submit_btn.click(warper.query, inputs = [msg, chatbot], outputs = [msg, chatbot])
  gr.close_all()
  demo.launch(server_name = FLAGS.host, server_port = FLAGS.port)

if __name__ == "__main__":
  add_options()
  app.run(main)

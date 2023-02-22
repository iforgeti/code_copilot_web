from flask import Flask, render_template, request
from utils.general import load_LSTM,generate
import torch
import torchtext




app = Flask(__name__)

model_path = "model/best-val-lstm_lm.pt"
params_path = "model/params.pt"
vocab_path ="model/vocab.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_LSTM(model_path,params_path).to(device)
vocab = torch.load(vocab_path)
tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
max_seq_len = 50
seed=0

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggestions')
def generate_suggestions():
    prompt = request.args.get('code', '')
    print(prompt)

    # real-----
    temperature = [0.4, 0.6, 0.8, 1.0]
    try:
        suggestions = [' '.join(generate(prompt.strip(), max_seq_len, temp, model, tokenizer, vocab, device, seed=seed))  for temp in temperature] 
    except:
        suggestions = []

    # html_text put to tag id suggestions
    suggestion_html = []
    for suggestion in suggestions:
        suggestion = suggestion.replace('\n', '<br>')
        suggestion_html.append(f'<li class="list-group-item">{suggestion}</li>')
    return {'suggestions': suggestion_html}


if __name__ == '__main__':
    app.run(debug=True)

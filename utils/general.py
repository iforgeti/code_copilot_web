from torch import nn
import torch
import torchtext
# models architect

vocab_path ="model/vocab.pt"


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
                
        super().__init__()
        
        self.num_layers = num_layers

        self.hid_dim   = hid_dim

        self.embedding = nn.Embedding(vocab_size,emb_dim)

        self.lstm = nn.LSTM(emb_dim,hid_dim,num_layers=num_layers,
                            dropout = dropout_rate, batch_first=True)
        # not do bidirectional (it only look forward)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hid_dim,vocab_size)



    def init_hidden(self, batch_size, device):
        # h0 have to be new not relate to any sentence
        # h0 just all 0
        #this function gonna be run in the beginning of the epoch
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        #this gonna run in every batch
        hidden, cell = hidden
        hidden = hidden.detach() #removing this hidden from gradients graph
        cell   = cell.detach()   #removing this cell from gradients graph
        return hidden, cell

    def forward(self, src, hidden):
        # h0 have to be new not relate to any sentence
        
        embed = self.embedding(src)
        # batch,embeddim

        output, hidden = self.lstm(embed, hidden) # hidden --> (h,c)
        # output - batch,seq,hiddim
        # hidden - num layer,batch,hiddim

        output = self.dropout(output)
        
        prediction = self.fc(output)
        # batch , seq , vocab

        return prediction, hidden

def load_LSTM(save_path = "model/best-val-lstm_lm.pt",params_path = "models/params.pt"):
    params = torch.load(params_path)

    model = LSTMLanguageModel( vocab_size=params["vocab_size"], emb_dim=params["emb_dim"], hid_dim=params["hid_dim"],
                               num_layers=params["num_layers"], dropout_rate=params["dropout_rate"])

    model.load_state_dict(torch.load(save_path))

    return model

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
    vocab = torch.load(vocab_path)
    model = load_LSTM()  
    max_seq_len = 50
    seed=0
    
    prompt = "import numpy as"
    temperatures = [0.4, 0.7, 1.0]
    gen_list = []

    for temp in temperatures:
        generation = generate(prompt, max_seq_len, temp, model, tokenizer, vocab, device, seed=seed)
        gen= ' '.join(generation)
        gen_list.append(gen)
        # html =. gen.replace('\n', '<br>')
        print(str(temp)+"\n"+gen+"\n")









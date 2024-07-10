
from utils import load_centralized_data
from model import SimCLRPredictor

from utils import NUM_CLASSES, useResnet18, tune_encoder, FINETUNE_EPOCHS

simclr_predictor = SimCLRPredictor(NUM_CLASSES, device, useResnet18 = useResnet18, tune_encoder = tune_encoder).to(device)


def evaluate_gb_model():
    load_model()
    
    trainloader, testloader = load_centralized_data()
    optimizer = torch.optim.Adam(self.simclr_predictor.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    fine_tune_predictor(trainloader, optimizer, criterion)
    
    return evaluate(testloader, optimizer, criterion)

    
def load_model():
    global simclr_predictor
    
    list_of_files = [fname for fname in glob.glob("./model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    
    simclr_predictor.load_state_dict(state_dict)
    
    return simclr_predictor


def fine_tune_predictor(trainloader, optimizer, criterion):
    simclr_predictor.train()

    for epoch in range(FINETUNE_EPOCHS):
        batch = 0
        num_batches = len(trainloader)
    

        for item in trainloader:
            (x, x_i, x_j), labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            outputs = self.simclr_predictor(x)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            # if batch % (print_interval * num_batches) == 0:
            print(f"Epoch: {epoch} Predictor Train Batch: {batch} / {num_batches}")

            
            batch += 1
            
            
def evaluate(testloader, optimizer, criterion):
    self.simclr_predictor.eval()
    
    total = 0

    correct = 0
    loss = 0

    batch = 0
    num_batches = len(self.testloader)

    with torch.no_grad():
        for item in self.testloader:
            (x, x_i, x_j), labels = item['img'], item['label']

            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            logits = self.simclr_predictor(x)
            values, predicted = torch.max(logits, 1)  
            
            total += labels.size(0)
            loss += self.criterion(logits, labels)
            correct += (predicted == labels).sum().item()

            print(f"Epoch: {epoch} Predictor Test Batch: {batch} / {num_batches}")
            
            batch += 1
  
        
    return loss / batch, correct / total

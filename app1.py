from typing import List
import numpy as np
import psycopg2
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from torch.utils.data import (DataLoader, Dataset)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, PhobertTokenizer, pipeline)
from arguments import load_args
from utils import *

app = FastAPI()


# Define a FastAPI dependency for the database connection
def get_db_cursor():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        yield cursor
    finally:
        cursor.close()
        conn.close()


args = load_args()
MODEL_PATH = args.model_pretrained
MAX_LEN = args.max_length
BATCH_SIZE = args.batch_size
NUM_LABEL = 4
device = torch.device(0)

# pipe = pipeline("text-classification", model="HieuAnh/phobert-travel")

tokenizer = AutoTokenizer.from_pretrained("HieuAnh/phobert-travel", do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained("HieuAnh/phobert-travel",
                                                           num_labels=NUM_LABEL,
                                                           output_attentions=False,
                                                           output_hidden_states=False)

model = model.to(device)


# fgfvvniufhsndguey
class Loader_testDataset(Dataset):
    def __init__(self, list_id, list_text, tokenizer, max_len):
        self.list_text = list_text
        self.list_id = list_id
        self.max_len = max_len
        self.text = list_text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_text)

    def __getitem__(self, index):
        text = self.text[index]
        idx = self.list_id[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )
        ids = inputs['input_ids'][0].to(device)
        mask = inputs['attention_mask'][0].to(device)

        return idx, {
            'input_ids': ids,
            'attention_mask': mask
        }


class text_sample(BaseModel):
    id: str
    text: str


class batch(BaseModel):
    list_item: List[text_sample]


AGENCY = '{105a72e6-7a82-47ad-b383-e46252ae95f3}'
SPAM = '{878aa7a3-8691-49b9-8018-2159a8b55175}'
SHARING = '{ac01344f-dd87-4064-8a3b-25561df9594f}'
POTENTIAL = '{bafe7c3e-106c-4ebd-89c8-27f64de0c668}'


# Define a function to connect to the database
def connect_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5433",
            database="postgres",
            user="postgres",
            password="W%2mN7&WkF")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


@app.post("/predict-batch")
async def predict_batch(cursor: psycopg2.extensions.cursor = Depends(get_db_cursor)):
    query = "SELECT contents.\"id\", contents.\"textContent\", contents.\"tagIds\" FROM smcc.contents"
    cursor.execute(query)
    print("Query executed successfully")
    rows = cursor.fetchall()
    print("Total rows: ", cursor.rowcount)

    # Filter rows with tagIds not equal to None
    rows = [row for row in rows if row[2] is None]

    input_data = [text_sample(id=row[0], text=row[1]) for row in rows]

    # print("Total rows after filtering: ", len(input_data))

    list_text = [data.text for data in input_data]
    list_text = [clean(text) for text in list_text]
    list_id = [data.id for data in input_data]
    id_preds = []
    label_preds = []
    prob_preds = []

    dataloader = Loader_testDataset(list_id, list_text, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(dataloader, BATCH_SIZE, shuffle=False)

    model.eval()
    for idx, batch in test_loader:
        with torch.no_grad():
            outputs = model(**batch)
            preds_labels = torch.sigmoid(outputs[0])
            preds_labels = preds_labels.detach().cpu().numpy()
            preds = [np.argmax(lb).item() for lb in preds_labels]
            probs = [np.max(lb).item() for lb in preds_labels]
            id_preds.extend(idx)
            label_preds.extend(preds)
            prob_preds.extend(probs)

    list_id = [id for id in id_preds]
    classification = {0: "AGENCY", 1: "PROTENTIAL", 2: "SHARING", 3: "SPAM"}
    probs = [str(round(i * 100)) for i in prob_preds]
    labels = [classification[i] for i in label_preds if i in classification]

    print("Total rows after predicting: ", len(list_id))

    # Update data in the database
    query_res = "UPDATE smcc.contents SET \"tagIds\" = %s WHERE \"id\" = %s"
    for i in range(len(list_id)):
        if labels[i] == "AGENCY":
            cursor.execute(query_res, (AGENCY, list_id[i]))
        elif labels[i] == "PROTENTIAL":
            cursor.execute(query_res, (POTENTIAL, list_id[i]))
        elif labels[i] == "SHARING":
            cursor.execute(query_res, (SHARING, list_id[i]))
        else:
            cursor.execute(query_res, (SPAM, list_id[i]))

    # Commit changes to the database
    cursor.connection.commit()
    # outputs = [[{"id": list_id[i]}, {"label": labels[i]}, {"probability": probs[i]}] for i in range(len(list_id))]
    # outputs = json.loads(json.dumps(outputs))
    # return outputs
    return {"message": "Update successful!"}


if __name__ == "__main__":
    uvicorn.run("app1:app")

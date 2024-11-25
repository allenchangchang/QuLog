import json
import pandas as pd

dataset_name = 'Thunderbird'

with open(f'{dataset_name}/embeddings.json', 'r') as f:
    embedding_json = json.load(f)

df = pd.read_csv(f'{dataset_name}/{dataset_name}.log_templates.csv')

template_event_dict = df.drop_duplicates(subset='EventTemplate').set_index('EventTemplate')['EventId'].to_dict()

new_embedding_dict = {}
for k, v in embedding_json.items():
    new_embedding_dict[template_event_dict[k]] = v

string = json.dumps(new_embedding_dict)
with open(f'{dataset_name}/new_embeddings.json', 'w') as f:
    f.write(string)

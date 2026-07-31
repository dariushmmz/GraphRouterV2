import wandb
import yaml
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from model.multi_task_graph_router import graph_router_prediction

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="configs/config.yaml")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--adaptive_updater", action="store_true")


args = parser.parse_args()
with open(args.config_file, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

wandb = None

data_dir = config['data_dir']

router_data_path = os.path.join(data_dir, 'router_training_data.csv')
llm_description_path = os.path.join(data_dir, 'LLM_Descriptions.json')
llm_embedding_path = os.path.join(data_dir, 'llm_description_embedding.pkl')

if config['feedback']:
    router_data_path = os.path.join(data_dir, 'feedback/router_training_data.csv')
    if not os.path.exists(router_data_path):
        print("[INFO] No feedback found")
        router_data_path = os.path.join(data_dir, 'router_training_data.csv')

graph_router_prediction(
    router_data_path=router_data_path,
    llm_path=llm_description_path,
    llm_embedding_path=llm_embedding_path,
    config=config, wandb=wandb,
    query_id=args.id, inference=True,
    adaptive=args.adaptive_updater
)

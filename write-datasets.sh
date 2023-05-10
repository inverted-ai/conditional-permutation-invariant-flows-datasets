output_path=output
interaction_path=path/to/interaction
python create_traffic_dataset.py --interaction_path $interaction_path --output_path $output_path --split train --prune_outer 0
python create_traffic_dataset.py --interaction_path $interaction_path --output_path $output_path --split val --prune_outer 0
python create_traffic_dataset.py --interaction_path $interaction_path --output_path $output_path --split train --prune_outer 1
python create_traffic_dataset.py --interaction_path $interaction_path --output_path $output_path --split val --prune_outer 1

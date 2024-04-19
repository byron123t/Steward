import os, csv, json


for file in os.listdir('data/results/mind2web'):
    interactables_true = 0
    interactables_total = 0
    limited_interactables_true = 0
    limited_interactables_total = 0
    correct_actions = 0
    total_actions = 0
    text_field_correct = 0
    text_field_total = 0
    print(file)
    if file.endswith('_elements.json'):
        data = json.load(open('data/results/mind2web/' + file, 'r'))
        for item in data['is_in_interactables']:
            interactables_total += 1
            if item:
                interactables_true += 1
        for item in data['is_in_limited_elements']:
            limited_interactables_total += 1
            if item:
                limited_interactables_true += 1
        print(interactables_true / interactables_total)
        print(limited_interactables_true / limited_interactables_total)
    elif file.endswith('_scores.json'):
        data = json.load(open('data/results/mind2web/' + file, 'r'))
        correct_actions += data['action_element_selection_match_close']
        total_actions += data['action_element_selection_total']
        text_field_correct += data['text_field_match']
        text_field_total += data['text_field_total']
        print(correct_actions / total_actions)
        print(text_field_correct / text_field_total)
        

for file in sorted(os.listdir('data/csvs')):
    if file.endswith('-scores.csv') and 'candidate_proposal' not in file:
        with open('data/csvs/' + file, 'r') as infile:
            reader = csv.DictReader(infile)
            next(reader)
            prior_correct = 0
            new_metric = 0
            new_correct = 0
            for i, row in enumerate(reader):
                if i > 0 and 'candidate_proposal' not in file:
                    break
                elif i > 0 and 'candidate_proposal' in file:
                    correct = int(float(row['correct']))
                    if int(row['episode']) < 25 and correct > prior_correct:
                        new_correct += 1
                    prior_correct = correct
            if 'candidate_action_selection' in file:
                metric = row['accuracy']
                metric_name = 'Top-1 Accuracy (Element + Action)'
            elif 'makes_sense' in file:
                metric = row['accuracy']
                metric_name = 'TPR (Makes Sense)'
            elif 'end_state_termination' in file:
                precision = float(row['tp']) / (float(row['tp']) + float(row['fp']))
                recall = float(row['tp']) / (float(row['tp']) + float(row['fn']))
                f1_score = 2 * (precision * recall) / (precision + recall)
                metric = f1_score
                metric_name = 'F1 Score (End State Termination)'
            elif 'secondary_parameter' in file:
                metric = row['accuracy']
                metric_name = 'TPR (Secondary Parameter)'
            elif 'candidate_proposal' in file:
                metric = new_correct / float(row['total'])
                # metric = row['accuracy']
                metric_name = 'Recall (Candidate Proposal)'

            print(file)
            print("{}:".format(metric_name), metric)
            print()
            print()
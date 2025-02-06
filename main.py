import re
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from scipy.stats import beta
from name_map import NAME_MAP, REPLACEMENTS
import pandas as pd
from nltk import pos_tag, word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')




def parse_time(time_str):
    try:
        return datetime.strptime(time_str, '%b %d, %Y %I:%M:%S %p')
    except ValueError:
        print(f"Warning: Unable to parse time string: {time_str}")
        return None

def analyze_message_data(file_path):
    date_pattern = r'^(\w{3} \d{2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)'
    phone_pattern = r'\+\d{11}'
    message_times = defaultdict(lambda: [0] * 24)
    total_messages_by_hour = [0] * 24
    word_counter = Counter()
    reactions_received = defaultdict(int)
    daily_message_counts = defaultdict(int)
    message_counts = defaultdict(int)
    reaction_matrix = defaultdict(lambda: defaultdict(int))
    name_mentions = defaultdict(lambda: defaultdict(int))
    current_sender = None
    current_time = None

    target_names = ["Tatum", "LeBron", "Bronny", "Brown", "Mahomes", "Maye"]



    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.replace(REPLACEMENTS[0], REPLACEMENTS[1]) # Fix Eli Email
            line = line.replace(REPLACEMENTS[2], REPLACEMENTS[3]) 
            line = line.strip()
            date_match = re.match(date_pattern, line)
            if date_match:
                current_time = parse_time(date_match.group(1))
                if current_time:
                    daily_message_counts[current_time.date()] += 1
                continue

            phone_match = re.match(phone_pattern, line)
            if phone_match:
                current_sender = NAME_MAP.get(phone_match.group(), phone_match.group())
                message_counts[current_sender] += 1
            elif current_sender and current_time:
                hour = current_time.hour
                message_times[current_sender][hour] += 1
                total_messages_by_hour[hour] += 1
                
                words = re.findall(r'\b\w+\b', line.lower())
                word_counter.update(word for word in words if len(word) > 3 and not word.isdigit())

                # Count name mentions
                for name in target_names:
                    if name.lower() in line.lower():
                        name_mentions[current_sender][name] += 1

            
            if line.startswith('Reactions:'):
                while True:
                    next_line = next(file, '').strip()
                    if not next_line or re.match(date_pattern, next_line):
                        break
                    reactor_match = re.search(phone_pattern, next_line)
                    if reactor_match:
                        reactor = NAME_MAP.get(reactor_match.group(), reactor_match.group())
                        if reactor != current_sender:  # Prevent self-reactions
                            reactions_received[current_sender] += 1
                            reaction_matrix[reactor][current_sender] += 1

    return message_times, total_messages_by_hour, word_counter, reactions_received, message_counts, reaction_matrix, name_mentions, daily_message_counts

def analyze_name_mentions_per_message(name_mentions, message_counts):
    target_names = ["Tatum", "LeBron", "Bronny", "Brown", "Mahomes", "Maye"]

    print("\nName Mention Analysis (per message):")
    for name in target_names:
        print(f"\nMentions of {name} per message:")
        mentions_per_message = [
            (sender, mentions[name] / message_counts[sender])
            for sender, mentions in name_mentions.items()
            if message_counts[sender] > 0
        ]
        sorted_mentions = sorted(mentions_per_message, key=lambda x: x[1], reverse=True)
        for sender, ratio in sorted_mentions:
            if ratio > 0:
                print(f"  {sender}: {ratio:.4f} mentions per message")

    # Find who mentioned each name the most per message
    print("\nWho mentioned each name the most per message:")
    for name in target_names:
        max_mentioner = max(
            ((sender, mentions[name] / message_counts[sender])
             for sender, mentions in name_mentions.items()
             if message_counts[sender] > 0),
            key=lambda x: x[1],
            default=(None, 0)
        )
        if max_mentioner[0]:
            print(f"{name}: {max_mentioner[0]} ({max_mentioner[1]:.4f} mentions per message)")
        else:
            print(f"{name}: No mentions")

def plot_time_distribution(message_times, total_messages_by_hour):
    plt.figure(figsize=(15, 10))
    hours = list(range(24))

    for sender, times in message_times.items():
        plt.plot(hours, times, label=sender, marker='o', alpha=0.7)

    plt.plot(hours, total_messages_by_hour, label='Total', linewidth=3, color='black', marker='s')

    plt.title('Message Distribution by Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Messages')
    plt.xticks(hours)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_reaction_heatmap(reaction_matrix):
    names = list(reaction_matrix.keys())
    data = np.array([[reaction_matrix[i][j] for j in names] for i in names])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=True, fmt='d', cmap='YlGnBu', xticklabels=names, yticklabels=names)
    plt.title('React Chart: Who Reacts to Whom')
    plt.xlabel('Receiver of Reaction')
    plt.ylabel('Giver of Reaction')
    plt.tight_layout()
    plt.show()

def plot_activity_calendar(daily_message_counts):
    # Convert the defaultdict to a pandas DataFrame
    dates = [date for date in daily_message_counts.keys()]
    values = list(daily_message_counts.values())
    df = pd.DataFrame({'date': dates, 'messages': values})
    
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year, month, and day
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Create a pivot table
    pivot_df = df.pivot(index='day', columns=['year', 'month'], values='messages')
    
    # Create the heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(pivot_df, cmap='YlOrRd', cbar_kws={'label': 'Number of Messages'})
    
    plt.title('Message Activity Calendar', fontsize=16)
    plt.xlabel('Month and Year')
    plt.ylabel('Day of Month')
    plt.tight_layout()
    plt.show()
def calculate_reaction_probabilities(reaction_matrix, message_counts, alpha=1, beta=1):
    names = list(reaction_matrix.keys())
    probabilities = {reactor: {} for reactor in names}
    
    for reactor in names:
        for receiver in names:
            reactions = reaction_matrix[reactor][receiver]
            messages = message_counts[receiver]
            
            # Use beta distribution to estimate probability
            a = reactions + alpha
            b = messages - reactions + beta
            probability = a / (a + b)  # Expected value of beta distribution
            
            probabilities[reactor][receiver] = probability
    
    return probabilities

def calculate_confidence(reactions, messages, confidence_level=0.95):
    a = reactions + 1
    b = messages - reactions + 1
    lower, upper = beta.interval(confidence_level, a, b)
    return upper - lower

def plot_reaction_probability_network(probabilities, reaction_matrix, message_counts, threshold=0.1, min_messages=10):
    G = nx.Graph()  # Changed to undirected graph
    names = list(probabilities.keys())
    
    # Add nodes
    for name in names:
        G.add_node(name)
    
    # Add edges
    for i, person1 in enumerate(names):
        for person2 in names[i+1:]:  # Avoid duplicate edges and self-loops
            prob1 = probabilities[person1][person2]
            prob2 = probabilities[person2][person1]
            reactions1 = reaction_matrix[person1][person2]
            reactions2 = reaction_matrix[person2][person1]
            messages1 = message_counts[person2]
            messages2 = message_counts[person1]
            
            # Calculate mutual probability
            mutual_prob = np.sqrt(prob1 * prob2)  # Geometric mean
            
            if (messages1 >= min_messages or messages2 >= min_messages) and mutual_prob > threshold:
                confidence1 = calculate_confidence(reactions1, messages1)
                confidence2 = calculate_confidence(reactions2, messages2)
                avg_confidence = (confidence1 + confidence2) / 2
                G.add_edge(person1, person2, weight=mutual_prob, confidence=avg_confidence)
    
    # Set up the plot
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    confidences = [G[u][v]['confidence'] for u, v in edges]
    
    # Normalize confidences for edge width
    max_confidence = max(confidences)
    normalized_confidences = [1 - (conf / max_confidence) for conf in confidences]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, 
                           edge_cmap=plt.cm.Reds, width=[5 * nc for nc in normalized_confidences])
    
    # Add edge labels
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f'Mutual Reaction Probability Network\n(Threshold: {threshold:.2f}, Min Messages: {min_messages})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_word_frequency_table(word_counter):
    # Filter words and get top 25

    # Filter for nouns only using NLTK
    nouns = {}
    for word, count in word_counter.items():
        if len(word) > 3 and not word.startswith('http'):
            pos = pos_tag([word])
            if pos[0][1].startswith('NN'):  # NN tags indicate nouns
                nouns[word] = count
                
    top_words = dict(sorted(nouns.items(), key=lambda x: x[1], reverse=True)[:25])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create table data
    table_data = [[word, count] for word, count in top_words.items()]
    
    # Create and customize table
    table = ax.table(cellText=table_data,
                    colLabels=['Word', 'Frequency'],
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.6, 0.4])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', weight='bold')
    
    plt.title('Top 25 Most Frequent Words', pad=20)
    plt.tight_layout()
    plt.show()


def analyze_name_mentions(name_mentions):
    target_names = ["Tatum", "LeBron", "Bronny", "Brown", "Mahomes", "Maye"]
    
    print("\nName Mention Analysis:")
    for name in target_names:
        print(f"\nMentions of {name}:")
        sorted_mentions = sorted([(sender, mentions[name]) for sender, mentions in name_mentions.items()], 
                                 key=lambda x: x[1], reverse=True)
        for sender, count in sorted_mentions:
            if count > 0:
                print(f"  {sender}: {count}")
    
    # Find who mentioned each name the most
    print("\nWho mentioned each name the most:")
    for name in target_names:
        max_mentioner = max(name_mentions.items(), key=lambda x: x[1].get(name, 0))
        if max_mentioner[1].get(name, 0) > 0:
            print(f"{name}: {max_mentioner[0]} ({max_mentioner[1][name]} times)")
        else:
            print(f"{name}: No mentions")

def analyze_most_reacted_messages(file_path):
    date_pattern = r'^(\w{3} \d{2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)'
    phone_pattern = r'\+\d{11}'
    
    messages = []
    current_sender = None
    current_time = None
    current_message = []
    reaction_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.replace(REPLACEMENTS[0], REPLACEMENTS[1])
            line = line.replace(REPLACEMENTS[2], REPLACEMENTS[3])
            line = line.strip()
            
            date_match = re.match(date_pattern, line)
            if date_match:
                # If we have a complete message, save it
                if current_message and current_sender and current_time:
                    messages.append({
                        'timestamp': current_time,
                        'sender': NAME_MAP.get(current_sender, current_sender),
                        'message': ' '.join(current_message),
                        'reaction_count': reaction_count
                    })
                
                current_time = parse_time(date_match.group(1))
                current_message = []
                reaction_count = 0
                continue
            
            phone_match = re.match(phone_pattern, line)
            if phone_match:
                current_sender = phone_match.group()
            elif line.startswith('Reactions:'):
                # Count reactions
                while True:                 
                    reaction_count += 1
                    next_line = next(file, '').strip()
                    if not next_line or re.match(date_pattern, next_line):
                        break
   
            elif current_sender and not line.startswith('Reactions:'):
                current_message.append(line)
    
    # Add the last message if exists
    if current_message and current_sender and current_time:
        messages.append({
            'timestamp': current_time,
            'sender': NAME_MAP.get(current_sender, current_sender),
            'message': ' '.join(current_message),
            'reaction_count': reaction_count
        })
    
    # Create DataFrame and sort by reactions
    df = pd.DataFrame(messages)
    df = df.sort_values('reaction_count', ascending=False)
    
    # Format timestamp
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def main():
    file_path = "WDGSN.txt"
    try:
        message_times, total_messages_by_hour, word_counter, reactions_received, message_counts, reaction_matrix, name_mentions, daily_message_counts = analyze_message_data(file_path)
        plot_time_distribution(message_times, total_messages_by_hour)
        plot_reaction_heatmap(reaction_matrix)
        plot_activity_calendar(daily_message_counts)

        plot_word_frequency_table(word_counter)
        print(f"\nTotal number of messages: {sum(message_counts.values())}")
        
        print("\nMessage count by person:")
        for sender, count in message_counts.items():
            print(f"{sender}: {count}")
        
        print("\n25 most common words (filtered):")
        for word, count in word_counter.most_common(25):
            print(f"{word}: {count}")
        
        print("\nReactions received and averages by person:")
        for sender, count in reactions_received.items():
            messages = message_counts.get(sender, 0)
            avg_reactions = count / messages if messages > 0 else 0
            print(f"{sender}: {count} reactions received, {messages} messages, {avg_reactions:.2f} avg reactions per message")
        
        if reactions_received:
            most_reacted = max(reactions_received, key=reactions_received.get)
            print(f"\nMost reactions garnered: {most_reacted} with {reactions_received[most_reacted]} total reactions")
        
        print("\nWho reacts to whom the most:")
        max_reactor = max(reaction_matrix.items(), key=lambda x: max(x[1].values()) if x[1] else 0)
        if max_reactor[1]:
            max_receiver = max(max_reactor[1].items(), key=lambda x: x[1])
            print(f"{max_reactor[0]} reacts most to {max_receiver[0]} with {max_receiver[1]} reactions")
        else:
            print("No reactions found in the data")
        
        most = analyze_most_reacted_messages(file_path).sort_values('reaction_count', ascending=False)

        most.to_csv('most.csv')
            # Calculate reaction probabilities
        reaction_probabilities = calculate_reaction_probabilities(reaction_matrix, message_counts)
        
        # Plot the new visualizations
        plot_reaction_probability_network(reaction_probabilities, reaction_matrix, message_counts, threshold=0.02, min_messages=3)
        analyze_name_mentions(name_mentions)
        analyze_name_mentions_per_message(name_mentions, message_counts)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
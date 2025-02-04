import re
from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx

# Name mapping
NAME_MAP = {
    '+16176865741': 'Eli',
    '+15086567672': 'Nikhil',
    '+17742495038': 'Nicket',
    '+18577565980': 'Easwer',
    '+16176783154': 'Svayam',
    '+16177947953': 'Arjun',
    '+17742850915': 'Neil',
    '+17742793396': 'Kyle',
    '+15033489354': 'Caden',
    '+18587521051': 'Carter', 
    '+12102842465': 'Jake',
    '+15033807803': 'Ian',
    '+15039708086': 'Natty'
}

def parse_time(time_str):
    try:
        return datetime.strptime(time_str, '%b %d, %Y %I:%M:%S %p')
    except ValueError:
        print(f"Warning: Unable to parse time string: {time_str}")
        return None

def identify_conversations(file_path, time_threshold=timedelta(minutes=10), participation_threshold=0.10):
    date_pattern = r'^(\w{3} \d{2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)'
    phone_pattern = r'\+\d{11}'
    conversations = []
    current_conversation = []
    conversation_participants = defaultdict(int)
    last_time = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.replace('elimendels55@gmail.com', '+16176865741')  # Fix Eli Email
            line = line.replace('Me', '+17742793396') 
            line = line.strip()
            
            date_match = re.match(date_pattern, line)
            if date_match:
                current_time = parse_time(date_match.group(1))
                if last_time and (current_time - last_time > time_threshold):
                    if current_conversation:
                        conversations.append((current_conversation, dict(conversation_participants)))
                    current_conversation = []
                    conversation_participants = defaultdict(int)
                last_time = current_time
                continue
            
            phone_match = re.match(phone_pattern, line)
            if phone_match:
                sender = NAME_MAP.get(phone_match.group(), phone_match.group())
                current_conversation.append(sender)
                conversation_participants[sender] += 1
    
    # Add the last conversation if it exists
    if current_conversation:
        conversations.append((current_conversation, dict(conversation_participants)))
    
    # Filter conversations based on participation threshold
    filtered_conversations = []
    for conv, participants in conversations:
        total_messages = sum(participants.values())
        significant_participants = {user for user, count in participants.items() 
                                    if count / total_messages >= participation_threshold}
        if significant_participants:
            filtered_conversations.append((conv, significant_participants))
    
    return filtered_conversations

def create_conversation_graph(conversations):
    G = nx.Graph()
    user_conversation_count = defaultdict(int)
    
    for conv, participants in conversations:
        for user1 in participants:
            user_conversation_count[user1] += 1
            for user2 in participants:
                if user1 < user2:  # Avoid duplicate edges
                    if G.has_edge(user1, user2):
                        G[user1][user2]['weight'] += 1
                    else:
                        G.add_edge(user1, user2, weight=1)
    
    # Add node attribute for conversation count
    nx.set_node_attributes(G, user_conversation_count, 'conversation_count')
    
    return G


def plot_conversation_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout for better spacing
    
    # Get conversation counts for node sizes
    conversation_counts = [G.nodes[node]['conversation_count'] for node in G.nodes()]
    
    # Normalize node sizes
    max_count = max(conversation_counts)
    node_sizes = [2000 * (count / max_count) for count in conversation_counts]
    
    # Draw nodes with sizes based on conversation count
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # Draw edges with weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    normalized_weights = [0.5 + 2 * (w / max_weight) for w in edge_weights]
    nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.5, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title("Conversation Interaction Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# The rest of your code remains the same

def main():
    file_path = "WHAT DEY GON SAY NOW LOG.txt"

    try:
        conversations = identify_conversations(file_path)
        G = create_conversation_graph(conversations)
        plot_conversation_graph(G)
        
        print(f"Total number of conversations: {len(conversations)}")
        print("\nNumber of conversations per user:")
        for node in G.nodes(data=True):
            print(f"{node[0]}: {node[1]['conversation_count']} conversations")
        
        print("\nTop 5 most frequent conversation pairs:")
        top_pairs = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]
        for user1, user2, data in top_pairs:
            print(f"{user1} and {user2}: {data['weight']} shared conversations")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
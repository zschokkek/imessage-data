import re

def clean_line(line):
    # Remove timestamps
    line = re.sub(r'\w{3} \d{2}, \d{4} \d{2}:\d{2}:\d{2} (AM|PM)', '', line)
    
    # Remove read receipts
    line = re.sub(r'\(Read by you after.*?\)', '', line)
    
    # Remove phone numbers
    line = re.sub(r'\+\d{11}', '', line)
    
    # Remove "This message responded to an earlier message."
    line = re.sub(r'This message responded to an earlier message\.', '', line)
    
    # Remove reactions
    line = re.sub(r'Reactions:.*', '', line)
    
    # Remove any leading/trailing whitespace
    line = line.strip()
    
    return line

def should_keep_line(line):
    # List of month names
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'Jul', 'July', 
              'August', 'Aug', 'Sep', 'September', 'October', 'November', 'December']
    
    # Check if line contains a month name and 'Twitter'
    contains_month = any(month in line for month in months)
    contains_twitter = 'by' in line or 'twitter' in line
    is_twitter_url = bool(re.match(r'https://twitter\.com/\w+/status/\d+(\?s=\d+(&t=\w+)?)?', line))
    
    # Keep the line if it doesn't contain both a month and 'Twitter', and is not a Twitter URL
    return not (contains_month or contains_twitter or is_twitter_url)


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if should_keep_line(line):
                cleaned_line = clean_line(line)
                if cleaned_line:  # Only write non-empty lines
                    outfile.write(cleaned_line + '\n')

# Paths
input_file = '/Users/kylezschokke/NEWEXPORT/chat_training_set.txt'
output_file = 'cleaned_chat_training_set.txt'

# Process the file
process_file(input_file, output_file)

print(f"Cleaned data has been saved to {output_file}")
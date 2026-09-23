import urllib.request
import csv
import io

url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
print(f"Downloading real dataset from {url}...")

try:
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    
    # Read the CSV
    reader = csv.reader(io.StringIO(content))
    header = next(reader)
    
    # find indices of 'text' and 'label'
    text_idx = header.index('text')
    label_idx = header.index('label')
    
    # Write to sample_data.csv
    with open('data/sample_data.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        
        count = 0
        for row in reader:
            if len(row) > max(text_idx, label_idx):
                writer.writerow([row[text_idx], row[label_idx]])
                count += 1
                if count >= 1000:  # let's save 1000 rows as sample data
                    break
                    
    print(f"Successfully downloaded and saved {count} rows to data/sample_data.csv")
except Exception as e:
    print(f"Error downloading dataset: {e}")

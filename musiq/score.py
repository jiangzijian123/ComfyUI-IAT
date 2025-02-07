def process_scores(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
        
        # Initialize variables
        scores = []
        name_score_list = []
        
        for line in data:
            # Strip whitespace and split each line
            line = line.strip()
            if not line:
                continue
            try:
                name, score = line.split(',')
                score = float(score)  # Convert to float
                scores.append(score)
                name_score_list.append((name, score))
            except ValueError:
                print(f"Invalid format: {line}")
        
        if not scores:
            return "No valid data found in the file!"
        
        # Calculate average score
        avg_score = sum(scores) / len(scores)
        
        # Sort the list by score in descending order
        sorted_scores = sorted(name_score_list, key=lambda x: x[1], reverse=True)
        
        # Get top 10 and bottom 10
        top_10 = sorted_scores[:10]
        bottom_10 = sorted_scores[-10:]
        
        return {
            "average_score": avg_score,
            "top_10": top_10,
            "bottom_10": bottom_10
        }
    
    except FileNotFoundError:
        return "File not found!"
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
file_path = "musiq_scores_fluxdrive.txt"  # Replace with your file path
result = process_scores(file_path)

print(file_path)
if isinstance(result, dict):
    print(f"Average score: {result['average_score']:.4f}")
    print("Top 10 highest scores:")
    for name, score in result['top_10']:
        print(f"  Name: {name}, Score: {score:.4f}")
    print("Bottom 10 lowest scores:")
    for name, score in result['bottom_10']:
        print(f"  Name: {name}, Score: {score:.4f}")
else:
    print(result)
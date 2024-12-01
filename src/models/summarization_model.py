import os
from transformers import pipeline


def summarizing_func(file_path, device, max_length=130, min_length=30, size_to_read=1800):
    summarizer = pipeline("summarization", model="jordiclive/flan-t5-3b-summarizer", device=device)
    
    summary_file = file_path + ".tmp" 
    #open transcription file
    with open(file_path, 'r') as f, open('summary.txt', 'w') as s:
        section = 1

        #Number of characters per section of input
        size_to_read = 1800
        section_content = f.read(size_to_read)

        #Iterate through remaining sections
        while len(section_content) > 0:
            s.write(f"Section {section}: \n")

            if len(section_content.split()) > 512:
                section_content = " ".join(section_content.split()[:512])
                
            adjusted_max_length = min(max_length, len(section_content.split()))

            # Call summarizer
            summary_list = summarizer(
                section_content,
                max_length=adjusted_max_length,
                min_length=min_length,
                do_sample=False
            )

            summary = summary_list[0]

            # Write section summary into summary file
            s.write(summary['summary_text'])

            # Try to read next section
            section_content = f.read(size_to_read)

            section += 1
            s.write("\n\n")

    # Replace the transcription file with the summary
    os.replace(summary_file, file_path)
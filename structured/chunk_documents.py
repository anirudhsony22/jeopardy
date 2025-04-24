import os
import pickle_utils as pkl
import argparse

def chunk_documents(input_dir='./database', output_base='./database', chunk_size=128, stride=64):
    doc_contents = pkl.open_pickle(os.path.join(input_dir, 'doc_contents.pkl'))
    doc_titles = pkl.open_pickle(os.path.join(input_dir, 'doc_titles.pkl'))

    if chunk_size==0:
        chunk_size = 'inf'

    chunked_contents = {}
    chunked_titles = {}
    chunked_metadata = {}
    new_doc_id = 0

    for original_id in sorted(doc_contents.keys()):
        content = doc_contents[original_id]
        title = doc_titles.get(original_id, f"Untitled {original_id}")
        tokens = content.strip().split()

        if chunk_size!='inf':
            for i in range(0, len(tokens) - chunk_size + 1, stride):
                chunk_tokens = tokens[i : i + chunk_size]
                chunk_text = ' '.join(chunk_tokens)

                chunked_contents[new_doc_id] = chunk_text
                chunked_titles[new_doc_id] = f"{title} [chunk {i//stride}]"
                chunked_metadata[new_doc_id] = {'original_doc_id': original_id, 'token_offset': i}

                new_doc_id += 1
        
        else:
            chunk_text = ' '.join(tokens)
            chunked_contents[new_doc_id] = chunk_text
            chunked_titles[new_doc_id] = title
            chunked_metadata[new_doc_id] = {'original_doc_id': original_id, 'token_offset': 0}
            new_doc_id += 1

        print(f"Chunks passed: {new_doc_id}")

    output_dir = os.path.join(output_base, str(chunk_size))
    os.makedirs(output_dir, exist_ok=True)

    pkl.save_pickle(chunked_contents, os.path.join(output_dir, 'doc_contents.pkl'))
    pkl.save_pickle(chunked_titles, os.path.join(output_dir, 'doc_titles.pkl'))
    pkl.save_pickle(chunked_metadata, os.path.join(output_dir, 'doc_metadata.pkl'))

    print(f"Chunking complete.")
    print(f"Total chunks: {new_doc_id}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk parsed documents into token-sized chunks")
    parser.add_argument('--chunk_size', type=int, default=128, help='Chunk size in tokens')
    parser.add_argument('--stride', type=int, default=64, help='Stride size for overlapping chunks')
    parser.add_argument('--input_dir', type=str, default='../database', help='Directory containing doc_contents.pkl')
    parser.add_argument('--output_dir', type=str, default='../database', help='Base output directory')
    args = parser.parse_args()

    chunk_documents(input_dir=args.input_dir, output_base=args.output_dir, chunk_size=args.chunk_size, stride=args.stride)

import datetime
import json
from dotenv import load_dotenv
import numpy as np
from angle_emb import AnglE
from openai import OpenAI
from input.prompts import Prompt, prompts
from data_classes import Context, Bio, TestCase

# Load the environment variables and set constants
load_dotenv()
openai_client = OpenAI()
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 50
NUM_RESPONSES = 3
TEST_CASES_FILE_PATH = "input/test_cases.json"
RESULTS_DIR_NAME = "results"


def embed_texts(texts: list[str]) -> np.ndarray:
    angle = AnglE.from_pretrained(
        'WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    embeddings = angle.encode(texts, to_numpy=True)
    np.save("./embs.npy", embeddings)  # TODO: Not sure what we want to do here
    return embeddings


# TODO: How do we want to compare the embeddings?
def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    '''
        Compares two embedding vectors and returns the cosine similarity.
        Output range is from -1 to 1, where 1 means the vectors are identical.
    '''
    return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))


def get_gpt_completion(prompt: str) -> list[str]:
    '''Makes a request to the OpenAI API and returns the response'''
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        n=NUM_RESPONSES
    )

    return [choice.message.content for choice in chat_completion.choices]


def execute_tests(prompt: Prompt, test_cases: list[TestCase]) -> dict[str, float]:
    '''Executes each test case against the given prompt and prints the score'''

    test_case_scores = {
        'prompt_id': prompt.id,
        'model': MODEL,
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS,
        'num_responses': NUM_RESPONSES,
        'scores': []
    }

    for test_case in test_cases:
        gpt_completions = get_gpt_completion(prompt.prompt.format(
            utterance=test_case.utterance,
            setting=test_case.context.setting,
            tone=test_case.context.tone,
            conversation_type=test_case.context.conversation_type,
            name=test_case.bio.name,
            age=test_case.bio.age,
            about_me=test_case.bio.about_me
        ))

        # Get the embeddings for the GPT responses and the good responses
        gpt_embeddings = embed_texts(gpt_completions)
        good_embeddings = embed_texts(test_case.good_completions)

        # Average the embeddings
        avg_gpt_embedding = np.mean(gpt_embeddings, axis=0)
        avg_good_embedding = np.mean(good_embeddings, axis=0)

        # Compare the embeddings and store the score
        cosine_similarity_score = cosine_similarity(
            avg_gpt_embedding, avg_good_embedding)
        test_case_scores['scores'].append({
            'test_case_id': test_case.id,
            'cosine_similarity_score': cosine_similarity_score
        })

    return test_case_scores


class CustomEncoder(json.JSONEncoder):
    '''Custom JSON encoder to handle numpy float32'''
    # TODO: Probably a much better way to handle this kind of thing

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


def main():
    # Load the test cases
    with open(TEST_CASES_FILE_PATH) as file:
        test_cases_json = json.load(file)
        test_cases = [TestCase(id=case['id'], utterance=case['utterance'], context=Context(
            **case['context']), bio=Bio(**case['bio']), good_completions=case['good_completions']) for case in test_cases_json]

    # Prepare the output file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_name = f"{RESULTS_DIR_NAME}/{timestamp}.json"
    with open(output_file_name, 'w') as file:
        json.dump({
            'results': [],
            'prompts': [p.to_dict() for p in prompts],
            'test_cases': [tc.to_dict() for tc in test_cases],
        }, file)

    # Execute each test case against each prompt and save the results
    for prompt in prompts:
        test_case_scores = execute_tests(prompt, test_cases)
        avg_cosine_similarity_score = np.mean(
            [score['cosine_similarity_score'] for score in test_case_scores['scores']])
        test_case_scores['average_cosine_similarity_score'] = avg_cosine_similarity_score
        with open(output_file_name, 'r+') as file:
            data = json.load(file)
            data['results'].append(test_case_scores)
            file.seek(0)
            json.dump(data, file, cls=CustomEncoder, indent=4, allow_nan=True)
            file.truncate()


if __name__ == "__main__":
    main()

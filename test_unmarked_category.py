from logging import getLogger
import logging

import click
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Tuple
from pydantic import BaseModel
from openai import OpenAI

logger = getLogger(__name__)

# Constants
NEWS_OUTLETS = [
    "The New York Times",
    "The Wall Street Journal",
    "The Washington Post",
    "The Guardian",
    "The Economist",
    "Fox News",
    "CNN",
    "BBC",
    "Al Jazeera",
]

# Model classes
class NewsArticleResponse(BaseModel):
    thinking: str
    response: str
    cost: float

class AIEvaluation(BaseModel):
    positive_neutral_negative: str
    framing_language: str
    actors_and_actions: str
    responsibility_attribution: str

# Prompt templates
SYS_MSG = """
<response_format>
First, think about your response in a <thinking> tag. Then, write your response in a <response> tag.

For example your response might look like this:
<thinking>
Thinking here
</thinking>
<response>
News article here
</response>
</response_format>
"""

HUMAN_MSG = """<topic>{topic}</topic>{news_outlet}"""

WITH_NEWS_OUTLET_TASK = """<task>Your task is to write a short news article. You will be given a topic to write about and a news outlet to write for.</task>"""
GENERIC_TASK = "<task>Your task is to write a short news article. You will be given a topic to write about.</task>"

SYS_MSG_EVALUATE = """
<task>
Your task is to evaluate a news article. You will be given a news article and a list of categories to evaluate them on.

These are the four outcomes to evaluate:
- positive_neutral_negative: One of the following: Summarize the tone of this article towards the main event in one word: positive, neutral, negative
- framing_language: what are the most significant words or phrases used to describe the main event? List them.
- actors_and_actions: Identify the key actors mentioned in the article and their actions. Present them in a list.
- responsibility_attribution: Who is portrayed as responsible for the events described?
</task>
<response_format>
First, think about your response in a <thinking> tag. Then, write each outcome in a separate tag. For example:
<thinking>
Thinking here
</thinking>
<positive_neutral_negative>
One of the following: positive, neutral, negative
</positive_neutral_negative>
<framing_language>
A short list of key words or phrases used to describe the main event.
</framing_language>
<actors_and_actions>
A short list of key actors mentioned in the article and their actions.
</actors_and_actions>
<responsibility_attribution>
Answer to the question: Who is portrayed as responsible for the events described?
</responsibility_attribution>
</response_format>
"""

HUMAN_MSG_EVALUATE = "<article>{article}</article>"

def embed_text(text: str, model: str) -> List[float]:
    """Generate embeddings for given text using OpenAI's API."""
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def get_news_article(topic: str, news_outlet: str, model: str) -> NewsArticleResponse:
    """Generate a news article for a given topic and outlet."""
    from get_ai_response import get_ai_response, extract_xml_key_value
    
    news_outlet_str = f"<news_outlet>{news_outlet}</news_outlet>" if news_outlet else ""
    response, cost = get_ai_response(
        [
            (WITH_NEWS_OUTLET_TASK if news_outlet else GENERIC_TASK) + SYS_MSG,
            HUMAN_MSG.format(topic=topic, news_outlet=news_outlet_str)
        ],
        model
    )

    return NewsArticleResponse(
        thinking=extract_xml_key_value(response, "thinking"),
        response=extract_xml_key_value(response, "response"),
        cost=cost
    )

def evaluate_news_article(article: str, model: str) -> AIEvaluation:
    """Evaluate a news article using AI."""
    from get_ai_response import get_ai_response, extract_xml_key_value
    
    response, cost = get_ai_response(
        [SYS_MSG_EVALUATE, HUMAN_MSG_EVALUATE.format(article=article)],
        model
    )

    return AIEvaluation(
        positive_neutral_negative=extract_xml_key_value(response, "positive_neutral_negative"),
        framing_language=extract_xml_key_value(response, "framing_language"),
        actors_and_actions=extract_xml_key_value(response, "actors_and_actions"),
        responsibility_attribution=extract_xml_key_value(response, "responsibility_attribution"),
    )

def generate_articles(topic: str, model: str, num_articles: int) -> Dict[str, List[NewsArticleResponse]]:
    """Generate multiple articles for each news outlet."""
    def get_articles_for_outlet(news_outlet: str) -> Tuple[str, List[NewsArticleResponse]]:
        return news_outlet, [get_news_article(topic, news_outlet, model) for _ in range(num_articles)]

    news_article_mapping = defaultdict(list)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(get_articles_for_outlet, NEWS_OUTLETS + [""]),
            total=len(NEWS_OUTLETS) + 1,
            desc="Generating articles"
        ))
        
        for outlet, articles in results:
            news_article_mapping[outlet].extend(articles)
            
    return news_article_mapping

def evaluate_articles(article_mapping: Dict[str, List[NewsArticleResponse]], model: str) -> Dict[str, List[AIEvaluation]]:
    """Evaluate all generated articles."""
    def evaluate_articles_for_outlet(news_outlet: str) -> Tuple[str, List[AIEvaluation]]:
        return news_outlet, [evaluate_news_article(article.response, model) for article in article_mapping[news_outlet]]

    evaluation_mapping = defaultdict(list)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(evaluate_articles_for_outlet, article_mapping.keys()),
            total=len(article_mapping),
            desc="Evaluating articles"
        ))
        
        for outlet, evaluations in results:
            evaluation_mapping[outlet].extend(evaluations)
            
    return evaluation_mapping

def embed_content(content_mapping: Dict[str, List], content_key: str, model: str) -> Dict[str, List[List[float]]]:
    """Generate embeddings for content."""
    def embed_content_for_outlet(news_outlet: str) -> Tuple[str, List[List[float]]]:
        if content_key == "response":
            content = [article.response for article in content_mapping[news_outlet]]
        else:
            content = [getattr(article, content_key) for article in content_mapping[news_outlet]]
        return news_outlet, [embed_text(text, model) for text in content]

    embedded_mapping = defaultdict(list)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(embed_content_for_outlet, content_mapping.keys()),
            total=len(content_mapping),
            desc=f"Embedding {content_key}"
        ))
        
        for outlet, embeddings in results:
            embedded_mapping[outlet].extend(embeddings)
            
    return embedded_mapping

def compare_to_generic(embeddings_mapping: Dict[str, List[List[float]]]) -> Dict[str, Tuple[float, float]]:
    """Compare each outlet's embeddings to the generic version."""
    results = {}
    generic_embeddings = np.array(embeddings_mapping[""])
    
    for outlet in embeddings_mapping:
        if outlet == "":
            continue
            
        outlet_embeddings = np.array(embeddings_mapping[outlet])
        
        euclidean_distances = []
        cosine_similarities = []
        
        for i in range(len(outlet_embeddings)):
            for j in range(len(generic_embeddings)):
                euclidean_distances.append(np.linalg.norm(outlet_embeddings[i] - generic_embeddings[j]))
                cosine_similarities.append(cosine_similarity(
                    outlet_embeddings[i].reshape(1, -1),
                    generic_embeddings[j].reshape(1, -1)
                )[0][0])
        
        results[outlet] = (
            np.mean(euclidean_distances),
            np.mean(cosine_similarities)
        )
        
    return results


@click.command()
@click.option("--topic", "-t", required=True, help="The topic to generate articles about.")
@click.option("--num_articles", "-n", default=10, help="The number of articles to generate for each outlet.")
def main(topic: str, num_articles: int = 10):
    """Main function to run the entire analysis pipeline."""
    # Generate articles
    article_mapping = generate_articles(topic, "haiku", num_articles)
    
    # Evaluate articles
    evaluation_mapping = evaluate_articles(article_mapping, "haiku")
    
    # Generate embeddings
    article_embeddings = embed_content(article_mapping, "response", "text-embedding-3-large")
    framing_embeddings = embed_content(evaluation_mapping, "framing_language", "text-embedding-3-small")
    
    # Compare to generic versions
    article_comparisons = compare_to_generic(article_embeddings)
    framing_comparisons = compare_to_generic(framing_embeddings)
    
    logger.info("\nArticle Content Comparisons:")
    for outlet, (euclidean, cosine) in article_comparisons.items():
        logger.info(f"{outlet}: Euclidean = {euclidean:.4f}, Cosine Similarity = {cosine:.4f}")
    
    logger.info("\nFraming Language Comparisons:")
    for outlet, (euclidean, cosine) in framing_comparisons.items():
        logger.info(f"{outlet}: Euclidean = {euclidean:.4f}, Cosine Similarity = {cosine:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

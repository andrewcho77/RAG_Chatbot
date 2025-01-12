from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()


def split_document(markdown_text):
    # Chunk the document based on h2 headers.
    headers_to_split_on = [("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    return md_header_splits


model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(
    model=model_name, pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get("PINECONE_CLOUD") or "aws"
region = os.environ.get("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-getting-started"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, dimension=embeddings.dimension, metric="cosine", spec=spec
    )
    # Wait for index to be ready
    # while not pc.describe_index(index_name).status["ready"]:
    # time.sleep(1)

namespace = "wondervector5000"

markdown_document = "## Introduction\n\nWelcome to the whimsical world of the WonderVector5000, an astonishing leap into the realms of imaginative technology. This extraordinary device, borne of creative fancy, promises to revolutionize absolutely nothing while dazzling you with its fantastical features. Whether you're a seasoned technophile or just someone looking for a bit of fun, the WonderVector5000 is sure to leave you amused and bemused in equal measure. Let's explore the incredible, albeit entirely fictitious, specifications, setup process, and troubleshooting tips for this marvel of modern nonsense.\n\n## Product overview\n\nThe WonderVector5000 is packed with features that defy logic and physics, each designed to sound impressive while maintaining a delightful air of absurdity:\n\n- Quantum Flibberflabber Engine: The heart of the WonderVector5000, this engine operates on principles of quantum flibberflabber, a phenomenon as mysterious as it is meaningless. It's said to harness the power of improbability to function seamlessly across multiple dimensions.\n\n- Hyperbolic Singularity Matrix: This component compresses infinite possibilities into a singular hyperbolic state, allowing the device to predict outcomes with 0% accuracy, ensuring every use is a new adventure.\n\n- Aetherial Flux Capacitor: Drawing energy from the fictional aether, this flux capacitor provides unlimited power by tapping into the boundless reserves of imaginary energy fields.\n\n- Multi-Dimensional Holo-Interface: Interact with the WonderVector5000 through its holographic interface that projects controls and information in three-and-a-half dimensions, creating a user experience that's simultaneously futuristic and perplexing.\n\n- Neural Fandango Synchronizer: This advanced feature connects directly to the user's brain waves, converting your deepest thoughts into tangible actions—albeit with results that are whimsically unpredictable.\n\n- Chrono-Distortion Field: Manipulate time itself with the WonderVector5000's chrono-distortion field, allowing you to experience moments before they occur or revisit them in a state of temporal flux.\n\n## Use cases\n\nWhile the WonderVector5000 is fundamentally a device of fiction and fun, let's imagine some scenarios where it could hypothetically be applied:\n\n- Time Travel Adventures: Use the Chrono-Distortion Field to visit key moments in history or glimpse into the future. While actual temporal manipulation is impossible, the mere idea sparks endless storytelling possibilities.\n\n- Interdimensional Gaming: Engage with the Multi-Dimensional Holo-Interface for immersive, out-of-this-world gaming experiences. Imagine games that adapt to your thoughts via the Neural Fandango Synchronizer, creating a unique and ever-changing environment.\n\n- Infinite Creativity: Harness the Hyperbolic Singularity Matrix for brainstorming sessions. By compressing infinite possibilities into hyperbolic states, it could theoretically help unlock unprecedented creative ideas.\n\n- Energy Experiments: Explore the concept of limitless power with the Aetherial Flux Capacitor. Though purely fictional, the notion of drawing energy from the aether could inspire innovative thinking in energy research.\n\n## Getting started\n\nSetting up your WonderVector5000 is both simple and absurdly intricate. Follow these steps to unleash the full potential of your new device:\n\n1. Unpack the Device: Remove the WonderVector5000 from its anti-gravitational packaging, ensuring to handle with care to avoid disturbing the delicate balance of its components.\n\n2. Initiate the Quantum Flibberflabber Engine: Locate the translucent lever marked “QFE Start” and pull it gently. You should notice a slight shimmer in the air as the engine engages, indicating that quantum flibberflabber is in effect.\n\n3. Calibrate the Hyperbolic Singularity Matrix: Turn the dials labeled 'Infinity A' and 'Infinity B' until the matrix stabilizes. You'll know it's calibrated correctly when the display shows a single, stable “∞”.\n\n4. Engage the Aetherial Flux Capacitor: Insert the EtherKey into the designated slot and turn it clockwise. A faint humming sound should confirm that the aetherial flux capacitor is active.\n\n5. Activate the Multi-Dimensional Holo-Interface: Press the button resembling a floating question mark to activate the holo-interface. The controls should materialize before your eyes, slightly out of phase with reality.\n\n6. Synchronize the Neural Fandango Synchronizer: Place the neural headband on your forehead and think of the word “Wonder”. The device will sync with your thoughts, a process that should take just a few moments.\n\n7. Set the Chrono-Distortion Field: Use the temporal sliders to adjust the time settings. Recommended presets include “Past”, “Present”, and “Future”, though feel free to explore other, more abstract temporal states.\n\n## Troubleshooting\n\nEven a device as fantastically designed as the WonderVector5000 can encounter problems. Here are some common issues and their solutions:\n\n- Issue: The Quantum Flibberflabber Engine won't start.\n\n    - Solution: Ensure the anti-gravitational packaging has been completely removed. Check for any residual shards of improbability that might be obstructing the engine.\n\n- Issue: The Hyperbolic Singularity Matrix displays “∞∞”.\n\n    - Solution: This indicates a hyper-infinite loop. Reset the dials to zero and then adjust them slowly until the display shows a single, stable infinity symbol.\n\n- Issue: The Aetherial Flux Capacitor isn't engaging.\n\n    - Solution: Verify that the EtherKey is properly inserted and genuine. Counterfeit EtherKeys can often cause malfunctions. Replace with an authenticated EtherKey if necessary.\n\n- Issue: The Multi-Dimensional Holo-Interface shows garbled projections.\n\n    - Solution: Realign the temporal resonators by tapping the holographic screen three times in quick succession. This should stabilize the projections.\n\n- Issue: The Neural Fandango Synchronizer causes headaches.\n\n    - Solution: Ensure the headband is properly positioned and not too tight. Relax and focus on simple, calming thoughts to ease the synchronization process.\n\n- Issue: The Chrono-Distortion Field is stuck in the past.\n\n    - Solution: Increase the temporal flux by 5%. If this fails, perform a hard reset by holding down the “Future” slider for ten seconds."
prepped_docs = split_document(markdown_document)
docsearch = PineconeVectorStore.from_documents(
    documents=prepped_docs,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)

# time.sleep(5)

index = pc.Index(index_name)
for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0],
        namespace=namespace,
        top_k=1,
        include_values=True,
        include_metadata=True,
    )

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.0,
)

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

query1 = "What are the first 3 steps for getting started with the WonderVector5000?"

query2 = "The Neural Fandango Synchronizer is giving me a headache. What do I do?"
answer1_with_knowledge = retrieval_chain.invoke({"input": query1})

print("Answer with knowledge:\n\n", answer1_with_knowledge["answer"])
print("\nContext used:\n\n", answer1_with_knowledge["context"])
print("\n")
time.sleep(2)

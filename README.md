# Scholarly Matchmaking: Finding Papers That Should Have Met

*A graph neural network approach to predicting missing citations in academic networks*

## 🎯 Project Overview

This project explores the hidden landscape of scholarly connections by using graph embeddings to predict likely citations between academic papers. Rather than claiming to identify "missing" citations objectively, this work functions as scholarly matchmaking—identifying papers that share enough intellectual DNA to warrant each other's attention.

Starting with a personal case study around my own eye-tracking research, this project demonstrates how graph neural networks can reveal patterns in knowledge networks that traditional search methods miss. The result is a data story that transforms abstract network analysis into practical insights about academic discovery and collaboration.

## 🧠 Core Concept

**The Analogy**: Just as dating apps suggest matches based on compatibility patterns rather than claiming objective perfection, this citation prediction model identifies papers with high probability of intellectual relevance based on learned network patterns.

**The Value**: Surfacing potentially valuable connections that researchers might want to explore, helping break down the invisible barriers between related work existing in parallel scholarly universes.

## 🏗️ Project Architecture

### Data Foundation
- **Source**: Academic citation network data collected via Semantic Scholar API
- **Storage**: Remote Neo4j Aura database (~500 papers, ~1000 relationships)
- **Scope**: Papers citing and cited by a 2009 eye-tracking research paper (personal case study)

> **Note**: Data collection and database population code is maintained in a separate repository. This project focuses on analysis, modeling, and storytelling using the pre-populated Neo4j database.

### Technical Stack
- **Graph Database**: Neo4j Aura (remote)
- **ML Framework**: PyTorch for TransE embeddings
- **Analysis**: NetworkX, pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn (portfolio-quality outputs)
- **Environment**: Jupyter notebooks
- **Deployment**: Local training → Remote model serving

## 📋 Database Schema

### Node Types
- **Paper**: `title`, `paperId`, `referenceCount`, `citationCount`, `publicationDate`
- **Author**: `name`, `id`, `url`, `paperCount`, `citationCount`, `hIndex`
- **PubVenue**: `name` (e.g., "Journal of Vision")
- **PubYear**: `year` (e.g., 2009)
- **Field**: `name` (e.g., "Computer Science")
- **Institution**: `name` (e.g., "The University of Melbourne")

### Relationships
- `CO_AUTHORED`: Author → Paper
- `PUBLISHED_IN`: Paper → PubVenue
- `PUB_YEAR`: Paper → PubYear
- `IS_ABOUT`: Paper → Field

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch networkx pandas matplotlib seaborn neo4j scikit-learn jupyter
```

### Environment Setup
Create a `.env` file with your Neo4j Aura credentials:
```
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=your-username
NEO4J_PASSWORD=your-password
```

### Quick Start
1. Clone this repository
2. Set up your environment variables
3. Launch Jupyter: `jupyter notebook`
4. Start with `01_network_exploration.ipynb`

## 📊 Analysis Pipeline

### Phase 1: Network Foundation
- **Data Extraction**: Query Neo4j → pandas DataFrames
- **Network Construction**: Build NetworkX graph representation
- **Baseline Analysis**: Node degrees, clustering, centrality metrics
- **Visualization**: Current "visible" citation network around seed paper

### Phase 2: Model Development
- **TransE Implementation**: PyTorch-based graph embedding model
- **Training Pipeline**: 80/20 train/test split with negative sampling
- **Evaluation**: Mean Reciprocal Rank (MRR), Hits@K metrics
- **Prediction Generation**: High-confidence missing link candidates

### Phase 3: Story Development 
- **Qualitative Analysis**: Manual examination of predicted connections
- **Before/After Visualization**: Network transformation reveal
- **Case Studies**: Deep dives into compelling predicted connections
- **Deployment Prep**: Model serving setup for remote access


## 🔧 Project Structure

```
├── notebooks/
│   ├── 01_network_exploration.ipynb    # EDA and baseline analysis
│   ├── 02_model_training.ipynb         # TransE implementation
│   ├── 03_prediction_analysis.ipynb    # Link prediction evaluation
│   └── 04_story_visualization.ipynb    # Final presentation
├── src/
│   ├── data_extraction.py              # Neo4j query utilities
│   ├── model.py                        # TransE model implementation
│   ├── evaluation.py                   # Performance metrics
│   └── visualization.py                # Plotting utilities
├── models/                             # Saved model checkpoints
├── outputs/                            # Generated visualizations
└── README.md
```

## 🚀 Deployment Strategy

- **Local Training**: Develop and train models in Jupyter environment
- **Model Serving**: Deploy trained embeddings for remote prediction API
- **Scalability**: Architecture supports expansion to larger citation networks

## 📝 Related Work

This project builds on established techniques in knowledge graph embeddings and citation network analysis, with a focus on practical application and interpretable results rather than state-of-the-art performance.

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome! Please open an issue or reach out directly.

---

*"The best way to understand a network is to try to predict it."* - This project applies that principle to the invisible web of scholarly knowledge.
# dp-100

## DP-100 Exam Preparation Material: Designing and Implementing a Data Science Solution on Azure

### Overview and Preparation Strategy

#### Exam Overview

**Certification Goal:** Validates your expertise in applying data science and machine learning to implement and run ML workloads on Azure, specifically using Azure Machine Learning (Azure ML).

**Exam Format:** Expect a mix of question types testing both conceptual understanding and practical application. Scenario-based questions will require you to choose the best Azure ML approach for a given situation.

**Duration & Passing Score:** Plan your time carefully during the exam. 700/1000 is the threshold, meaning you need a strong grasp across most domains.

**Languages:** Primarily English, but check Pearson VUE for available translations.

#### Preparation Strategy and Prerequisites

**Background Knowledge:** Solid Python programming (especially libraries like Pandas, Scikit-learn), understanding of ML concepts (regression, classification, clustering, model evaluation), and familiarity with cloud basics are essential. Knowing Azure fundamentals (resource groups, storage) is beneficial (AZ-900 knowledge helps but isn't strictly required).

**Study Plan:** Consistency is key. Break down the syllabus (using this guide!), allocate time for each domain. Prioritize understanding concepts first, then reinforce with hands-on labs using the Azure ML Python SDK v2 and the Studio interface. Finish with practice tests to gauge readiness and identify weak spots.

**Resources:**

- **Microsoft Learn:** The official path (DP-100: Designing and implementing a data science solution on Azure) is your primary resource. Pay close attention to modules updated for SDK v2.
- **Azure ML Documentation:** Use this for deep dives into specific features, classes, methods, and troubleshooting.
- **Hands-on Labs:** The MS Learn labs are crucial. Also, explore the Azure ML examples GitHub repository (github.com/Azure/azureml-examples).
- **Practice Exams:** Use MeasureUp (official) or other reputable providers after you feel confident with the material and labs. They help simulate exam pressure and question style.

### Detailed Syllabus Breakdown

#### Domain 1: Design and Prepare a Machine Learning Solution (20–25%)

**A. Designing a Machine Learning Solution**

**Dataset Considerations:**

- **Concept:** Before training, understand your data. What format is it in (CSV, JSON, Parquet, images, text)? Where does it come from (Azure Blob, Data Lake, SQL DB)? How large is it? Is it structured or unstructured?
- **Importance:** Dictates storage choices (datastore types), data ingestion methods (SDK commands, UI uploads, Azure Data Factory), and preprocessing steps. Data quality (missing values, outliers, consistency) directly impacts model performance. Define a clear schema.
- **Azure ML Context:** Use Azure ML Datastores to connect to storage services and Data Assets (v2 SDK) / Datasets (v1 SDK/UI) to represent and manage your data within the workspace for versioning and reproducibility.

**Compute Specifications:**

- **Concept:** ML tasks require computational power. Different tasks have different needs (CPU for traditional ML, GPU for deep learning, high memory for large datasets). Scalability is important for handling varying loads or parallel processing.
- **Importance:** Choosing the right compute prevents bottlenecks, controls costs, and ensures timely completion of tasks. Under-provisioning leads to slow training; over-provisioning wastes money.
- **Azure ML Context:** Azure ML offers various Compute Targets:
  - **Compute Instance:** A managed, cloud-based workstation for development (like a Jupyter Notebook server). Good for interactive exploration and small training jobs.
  - **Compute Cluster:** Scalable clusters of VMs (CPU or GPU) for training, batch inference, and parallel tasks. Auto-scaling helps manage costs.
  - **Attached Compute:** Link existing resources like Azure Databricks or Azure Synapse Spark pools (less common for core training in DP-100 scope, but good to know exists).
  - **Serverless Compute (Preview/Newer):** Azure handles the compute infrastructure automatically for specific job types (focus on Clusters/Instances for the exam).

**Development Approach:**

- **Concept:** How will you build your ML solution? Will you write Python code using SDKs, or use a graphical interface?
- **Importance:** Choice depends on team skills, project complexity, and desired control level.
- **Azure ML Context:**
  - **Code-First (Python SDK v2):** Offers maximum flexibility and control. Ideal for data scientists comfortable with Python. Allows integration with standard dev tools (Git, VS Code) and automation. This is heavily tested.
  - **Low-Code/No-Code (Azure ML Studio/Designer):** Visual drag-and-drop interface (Designer) or guided UI (AutoML in Studio). Good for rapid prototyping, citizen data scientists, or standard tasks. Understand the capabilities and limitations of both.

**Exercises (Domain 1.A):**

- **Scenario Analysis:** Given a scenario (e.g., "Train a model to predict customer churn using a 10GB CSV file stored in Azure Blob Storage, requiring GPU acceleration for a deep learning model"), identify:
  - Appropriate Azure ML Data Asset type/creation method.
  - Recommended Compute Target(s) for training.
  - Whether SDK or Designer would be more suitable and why.
- **Resource Planning:** For a project involving image classification on 100,000 images, outline the considerations for choosing compute (CPU vs. GPU, cluster size, scaling).

**B. Creating and Managing an Azure Machine Learning Workspace**

**Workspace Creation and Management:**

- **Concept:** The Azure ML Workspace is the central hub for all your ML activities. It's an Azure resource that organizes everything else.
- **How-to:** Create via Azure Portal (UI walk-through), Azure CLI (az ml workspace create ...), or Python SDK v2 (MLClient.workspaces.begin_create_or_update(...)).
- **Architecture:** Understand that a Workspace relies on associated resources created automatically (or linked): Azure Storage Account (default datastore, stores notebooks, artifacts), Azure Key Vault (secrets management), Azure Application Insights (monitoring, logging), Azure Container Registry (Docker images for environments). Know these dependencies.

**Data Management:**

- **Datastores:** Links to Azure storage services (Blob, Data Lake Gen1/Gen2, Files, SQL DB, etc.). They don't move data, just store connection info securely. Create/register via Studio UI, SDK (Datastore class, ml_client.datastores.create_or_update), or CLI (az ml datastore create ...).
- **Data Assets (v2):** Abstractions over your data (files, folders, tables). Key features: versioning, tracking, sharing. Represent data sources used in jobs/pipelines. Create from local files, datastores, or public URLs using SDK (Data class, ml_client.data.create_or_update) or Studio UI. Understand types: uri_file, uri_folder, mltable.

**Compute Resources:**

- **Creation/Management:** Create Compute Instances and Compute Clusters via Studio UI, SDK (ComputeInstance, AmlCompute classes, ml_client.compute.begin_create_or_update), or CLI (az ml compute create ...).
- **Configuration:** Specify VM size, min/max nodes (for clusters), idle shutdown times (cost saving!), SSH access, virtual network integration.
- **Monitoring/Cost:** Monitor utilization via Azure Monitor and the Studio interface. Use tags for cost tracking. Implement auto-scaling and idle shutdown for clusters to manage costs effectively.

**Source Control Integration:**

- **Concept:** Essential for reproducibility, collaboration, and CI/CD.
- **How-to:** Azure ML integrates with Git repositories (GitHub, Azure Repos). You can clone repos onto a Compute Instance terminal or directly within VS Code connected to the instance. Jobs can be configured to pull code from a specific repo/commit.

**Asset Sharing:**

- **Environments:** Define the runtime context for jobs (Python version, conda/pip packages, Docker settings). Create from Docker images, build contexts, or conda YAML files. Promotes reproducibility. Manage via SDK (Environment class, ml_client.environments.create_or_update), CLI (az ml environment create), or Studio UI. Share environments within the workspace or via registries.
- **Registries:** A way to share assets (models, components, environments) across multiple workspaces, enabling reuse and standardization within an organization.

**Exercises (Domain 1.B):**

- **Workspace Setup (SDK):** Write a Python script using SDK v2 (azure-ai-ml) to:
  - Connect to your Azure subscription.
  - Create a new Azure ML Workspace in a specific resource group and location.
  - Verify the creation of associated resources (Storage, Key Vault, etc.) in the Azure Portal.
- **Data Registration (Studio & SDK):**
  - Upload a sample CSV file to the default Blob storage associated with your workspace via the Azure Portal.
  - Using Studio: Create a Datastore connection to this Blob container (if not default). Then, create a Data Asset (type uri_file or mltable) representing the uploaded CSV. Note its version.
  - Using SDK: Write Python code to register the same Blob container as a Datastore and then create a new version of the Data Asset pointing to the CSV.
- **Compute Creation (CLI):** Use the Azure CLI (az ml compute create) to create:
  - A small CPU-based Compute Cluster (type: amlcompute) with min 0, max 2 nodes, and an idle shutdown time of 30 minutes.
  - A Standard DS3 v2 Compute Instance with SSH access enabled.
- **Environment Definition (SDK):** Create an Azure ML Environment using the SDK v2 that specifies Python 3.9 and includes scikit-learn, pandas, and mlflow. Register it in your workspace.

#### Domain 2: Explore Data and Run Experiments (20–25%)

**A. Automated Machine Learning (AutoML)**

**Concept:** Automates the time-consuming, iterative tasks of ML model development. Tries different algorithms, preprocessing steps, and hyperparameters to find the best model for your data and task.

**Supported Task Types:**

- **Tabular Data:** Classification (binary/multi-class), Regression, Forecasting.
- **Computer Vision:** Image Classification (multi-class/multi-label), Object Detection, Instance Segmentation.
- **Natural Language Processing (NLP):** Text Classification (multi-class/multi-label), Named Entity Recognition (NER).

**Training Options:**

- **Configuration:** Define the primary metric (e.g., AUC, Accuracy, R2 Score, NDCG), exit criteria (training time, score threshold), allowed/blocked algorithms, concurrency limits.
- **Preprocessing:** AutoML automatically handles missing data imputation, feature scaling, encoding categorical features, etc. (featurization). You can customize some aspects.
- **Responsible AI:** View Responsible AI dashboards for the best AutoML models (interpretability, fairness analysis – especially for tabular data).
- **Evaluation:** Review the list of models tried, their performance metrics, preprocessing steps applied, and generated code (for tabular). Select the best performing model based on your primary metric and other business requirements. Deploy directly or register the model.
- **How-to:** Run via Studio UI (guided wizard) or Python SDK v2 (automl module, e.g., automl.classification(...)).

**Exercises (Domain 2.A):**

- **AutoML Run (Studio):** Using a sample tabular dataset (e.g., diabetes prediction, available in Azure ML samples), configure and run an AutoML Classification job via the Azure ML Studio UI.
  - Select the dataset (Data Asset).
  - Choose the target column.
  - Select compute.
  - Define the primary metric (e.g., AUC_weighted).
  - Set an experiment timeout (e.g., 20 minutes).
  - Review the results, identify the best model, and explore its details and metrics. Explore the generated Responsible AI insights.
- **AutoML Run (SDK):** Repeat the above exercise using the Python SDK v2.
  - Load the data asset.
  - Configure the automl.classification job (specify task type, primary metric, training data, target column name, compute target, limits).
  - Submit the job (ml_client.jobs.create_or_update(automl_job)).
  - Monitor the run and retrieve the best model details programmatically.

**B. Custom Model Training with Notebooks**

**Interactive Development:**

- **Concept:** Use familiar tools like Jupyter Notebooks or VS Code (with Azure ML extension) for iterative data exploration, feature engineering, and model training.
- **Azure ML Context:** Run notebooks directly on a Compute Instance. This provides access to the workspace, datastores, compute, and SDK within a managed environment.
- **Data Access:** Use the SDK v2 (ml_client.data.get(...) to get path, Input class for jobs) to access Data Assets. Load data into Pandas DataFrames or other structures for wrangling.
- **Feature Stores:** (More advanced, but good to know) Azure ML Feature Store allows defining, managing, and serving features consistently across different models and teams, preventing feature drift and redundant work. You can retrieve pre-computed features from a feature store for training.

**Compute Instance Configuration:**

- **Concept:** Your personal development server in the cloud.
- **Configuration:** Choose VM size, enable/disable SSH, set up idle shutdown, assign to a user. Comes pre-installed with common data science tools and the Azure ML SDK.
- **Terminal Access:** Use the integrated terminal (in Jupyter/JupyterLab/VS Code) or SSH (if enabled) for installing custom packages, managing files, or running scripts.

**Experiment Tracking:**

- **Concept:** Recording parameters, metrics, code versions, artifacts (like model files or plots), and logs associated with each training run. Crucial for reproducibility, comparison, and debugging.
- **Azure ML Context:** Azure ML automatically integrates with MLflow. Use standard MLflow logging commands (mlflow.log_param, mlflow.log_metric, mlflow.log_artifact, mlflow.sklearn.autolog) within your Python training script. Runs are tracked as Jobs (formerly Experiments/Runs in v1) in the Azure ML Studio.
- **Evaluation:** Access job details in the Studio UI or via SDK (ml_client.jobs.get, ml_client.jobs.stream) to view logs, metrics, parameters, outputs, and compare different runs.

**Exercises (Domain 2.B):**

- **Notebook Setup:**
  - Create a Compute Instance in your workspace.
  - Access the Jupyter or VS Code interface on the instance.
  - Clone a sample training notebook repository (e.g., from azureml-examples) using the terminal.
- **Interactive Training:**
  - Open a sample Python training notebook (e.g., basic scikit-learn classification).
  - Modify the notebook to:
    - Load data from a registered Azure ML Data Asset (using Input or ml_client.data.get).
    - Add MLflow logging for at least one parameter (e.g., learning_rate) and one metric (e.g., accuracy).
    - Use mlflow.sklearn.autolog() for automatic logging.
  - Run the notebook cells interactively on the Compute Instance.
- **Experiment Review:**
  - Navigate to the "Jobs" section in Azure ML Studio.
  - Find the job corresponding to your notebook execution.
  - Examine the logged parameters, metrics, artifacts (like pickled models if autologged), and outputs/logs.

**C. Hyperparameter Tuning**

**Concept:** Finding the optimal set of hyperparameters (values set before training, like learning rate, number of trees) for a model to maximize its performance on a specific metric.

**Tuning Strategies (Sampling Methods):**

- **Grid Sampling:** Tries every possible combination of the specified hyperparameter values. Exhaustive but computationally expensive.
- **Random Sampling:** Randomly selects combinations from the defined search space. More efficient than grid, often finds good results faster.
- **Bayesian Sampling:** Uses results from previous runs to intelligently choose the next hyperparameter combination to try. Often the most efficient, balancing exploration and exploitation. (Uses Hyperopt library).

**Configuration:**

- **Search Space:** Define the range or discrete values for each hyperparameter to explore (e.g., learning_rate = uniform(0.01, 0.1), n_estimators = choice(100, 200, 500)). Use azure.ai.ml.sweep functions (choice, uniform, loguniform, etc.).
- **Primary Metric:** The metric the tuning process aims to optimize (maximize or minimize, e.g., accuracy, RMSE). Must be logged by your training script using MLflow.
- **Early Termination:** Stop poorly performing runs early to save compute. Policies include:
  - **Bandit Policy:** Stops runs whose primary metric falls outside a specified slack factor/amount compared to the best run so far.
  - **Median Stopping Policy:** Stops runs whose best metric is worse than the median of running averages across all runs.
  - **Truncation Selection Policy:** Cancels the lowest performing X% of runs at each evaluation interval.
- **How-to:** Define a Sweep Job using the Python SDK v2 (SweepJob class) or CLI (az ml job create --type sweep). This job wraps your command/script job and manages the tuning process.

**Exercises (Domain 2.C):**

- **Prepare Training Script:** Ensure you have a Python training script (e.g., the one used in 2.B) that:
  - Accepts hyperparameters as command-line arguments (argparse).
  - Logs the primary metric using mlflow.log_metric().
- **Define Sweep Job (SDK):** Write a Python script using SDK v2 to create a SweepJob:
  - Define the command job it will run (your training script).
  - Define the search_space using functions like choice, uniform.
  - Specify the sampling_algorithm (e.g., 'random' or 'bayesian').
  - Define the objective (goal: 'maximize' or 'minimize', primary_metric: name of the logged metric).
  - Configure an early_termination policy (e.g., MedianStoppingPolicy).
  - Set limits (e.g., max_total_trials, max_concurrent_trials).
  - Specify the compute target.
- **Run and Analyze:** Submit the sweep job (ml_client.jobs.create_or_update(sweep_job)). Monitor its progress in Studio. Once completed, identify the best trial and its corresponding hyperparameter values and performance.

#### Domain 3: Train and Deploy Models (25–30%)

**A. Running Model Training Scripts**

**Job Submission:**

- **Concept:** Execute your training script as a non-interactive Job on Azure ML compute.
- **Data Consumption:** Use the Input class in your SDK job definition to link Data Assets to your script. Azure ML handles mounting or downloading the data to the compute target. Access the data path within your script (often provided via environment variables or command-line args). Common input modes are ro_mount, download, direct.
- **Job Configuration:** Define the job using SDK v2 (command function or CommandJob class) or YAML files (for CLI/SDK). Specify:
  - **code:** Path to your training script/folder.
  - **command:** The exact command to execute (e.g., python train.py --data_path ${{inputs.training_data}} --learning_rate 0.01). Use ${{inputs.<input_name>}} and ${{outputs.<output_name>}} for data binding.
  - **inputs:** Dictionary mapping input names (used in command) to Input objects (Data Assets).
  - **outputs:** Dictionary mapping output names to Output objects (specifying where to store output like trained models, often using type uri_folder or mlflow_model).
  - **environment:** The registered Azure ML Environment or an inline definition.
  - **compute:** Name of the Compute Cluster or Serverless target.
  - **display_name, experiment_name:** For organization.
- **Submission:** Use ml_client.jobs.create_or_update(job) (SDK) or az ml job create -f <yaml_file> (CLI).

**Troubleshooting:**

- **Logs:** Access job logs (stdout, stderr) via Studio UI ("Outputs + logs" tab) or SDK (ml_client.jobs.get(job_name).studio_url, ml_client.jobs.stream(job_name)). Check user_logs/std_log.txt (for print statements, errors) and system_logs (for Azure ML infrastructure issues).

**Tracking Runs (MLflow):**

- **Concept:** As mentioned before, MLflow is integrated. Jobs automatically start an MLflow run.
- **Usage:** Use mlflow.log_param, mlflow.log_metric, mlflow.log_artifact, mlflow.save_model within your training script. The Output(type='mlflow_model') in the job definition simplifies saving models in the MLflow format.
- **Benefit:** Centralized tracking of all job details, metrics, parameters, and model artifacts within the Azure ML workspace.

**Exercises (Domain 3.A):**

- **Create Command Job (SDK/YAML):**
  - Take your hyperparameter-tuned training script (from 2.C exercise).
  - Define a CommandJob using the SDK v2 (or create a corresponding YAML file):
    - Reference the training script folder (code).
    - Set the command to run the script, passing fixed, optimal hyperparameters found during tuning, and input/output paths using ${{...}} syntax.
    - Define an Input pointing to your training Data Asset.
    - Define an Output of type mlflow_model for the trained model.
    - Specify the Environment created earlier.
    - Specify your Compute Cluster.
- **Submit and Monitor:** Submit the job using SDK or CLI. Monitor its status.
- **Verify Output:** Once completed, check the Job details in Studio:
  - Verify the parameters and metrics logged via MLflow.
  - Confirm the model was saved as an artifact in the MLflow format under the "Outputs + logs" or directly linked in the job overview if using mlflow_model output.
  - Inspect the logs (user_logs/std_log.txt) for any print statements or errors.

**B. Building and Implementing Training Pipelines**

**Pipeline Construction:**

- **Concept:** Chain multiple ML tasks (data prep, training, validation, registration) into a reproducible and manageable workflow. Each step is a Component.
- **Components:** Reusable, self-contained pipeline steps. Can be built from:
  - **A Python script (command component):** Define inputs, outputs, command, environment. Similar to a Command Job but designed for pipeline use.
  - **Registered components:** Reuse components defined by you or others.
- **Pipeline Definition (SDK v2):** Use the @dsl.pipeline decorator on a Python function. Inside the function:
  - Load components (from YAML files or registered components: load_component).
  - Instantiate components, passing parameters and binding outputs of one step to inputs of the next (e.g., train_step.outputs.model_output -> eval_step.inputs.model_input).
  - Define pipeline inputs/outputs.
- **Data Flow:** Data Assets or intermediate data flow between components via specified inputs and outputs.

**Scheduling and Monitoring:**

- **Scheduling:** Trigger pipelines automatically based on a time schedule (e.g., daily, weekly) or data changes (e.g., new data arriving in a datastore - event-based trigger). Set up via Studio UI or SDK. Crucial for MLOps (retraining).
- **Monitoring:** Monitor pipeline runs (parent Pipeline Job) and individual component jobs in the Studio UI. Check statuses, logs, outputs for each step. Troubleshoot failures by examining the logs of the failed component job.

**Exercises (Domain 3.B):**

- **Create Components (YAML/SDK):**
  - Create separate Python scripts for:
    - Data Preprocessing (takes raw data path, outputs processed data path).
    - Model Training (takes processed data path, hyperparameters, outputs model path).
  - Define corresponding Azure ML Command Components for each script using YAML files (or SDK command_component function). Define their respective inputs and outputs clearly.
- **Build Pipeline (SDK):** Write a Python script using @dsl.pipeline to:
  - Load the two components created above.
  - Define pipeline inputs (e.g., raw data location, learning rate).
  - Instantiate the components, connecting the output of the prep component to the input of the training component.
  - Configure the compute target for the components.
- **Run and Schedule:**
  - Compile and submit the pipeline (ml_client.jobs.create_or_update(pipeline_job)).
  - Monitor the run in Studio, observing the graph and the status of individual component jobs.
  - (Optional) Set up a time-based schedule for this pipeline via the Studio UI's "Pipelines" -> "Published pipelines" section (after publishing the pipeline).

**C. Model Management**

**Model Registration:**

- **Concept:** Storing and versioning your trained models in the Azure ML Workspace Model Registry. Makes models discoverable, manageable, and ready for deployment.
- **How-to:**
  - **From Job Output:** If your training job/pipeline saves a model in MLflow format (using mlflow.save_model or Output(type='mlflow_model')), you can register it directly from the job output in the Studio UI or using SDK (ml_client.models.create_or_update with path=job.outputs.<output_name>).
  - **MLflow:** Use mlflow.register_model("runs:/<run_id>/path/to/model", "model_name") within your script or after a run.
  - **SDK/CLI:** Manually register a model file/folder using ml_client.models.create_or_update (SDK) or az ml model create (CLI), specifying path, name, version, type (mlflow_model, custom_model, etc.).
- **Model Signature:** (Especially for MLflow models) Defines the expected input schema and output schema of the model. Helps ensure compatibility during deployment. Often inferred automatically by mlflow.save_model.
- **Feature Retrieval Specs:** (Advanced) Package information about required features alongside the model, potentially linking to a feature store for automated retrieval during inference.

**Assessment and Responsible AI (RAI):**

- **Concept:** Evaluating models beyond just accuracy. Includes fairness (bias detection/mitigation across sensitive groups), explainability/interpretability (understanding why a model makes predictions), error analysis, causal analysis, and model performance.
- **Azure ML Tools:** Generate Responsible AI dashboards (often as part of AutoML or via dedicated components/SDK functions - azureml-responsibleai package). Analyze these dashboards to understand model behavior, identify potential biases, and ensure the model aligns with ethical guidelines and requirements. Use techniques like SHAP or LIME for interpretability.

**Exercises (Domain 3.C):**

- **Register Model (SDK/Studio):**
  - Find the completed training job (from 3.A or 3.B) that produced an mlflow_model output.
  - Using SDK: Write code to get the job output path and register the model using ml_client.models.create_or_update. Specify a name, description, and potentially tags.
  - Using Studio: Navigate to the job, find the model output artifact, and use the "Register model" button.
  - Verify the model appears in the "Models" section of your workspace. Note its name and version.
- **Generate RAI Dashboard (If applicable):**
  - If you ran an AutoML job (tabular), revisit the best run and explore its "Responsible AI" tab in detail.
  - (Advanced) Explore MS Learn labs or examples on generating a custom RAI dashboard for a scikit-learn model using the responsibleai SDK/components. Analyze the fairness and interpretability results.

**D. Model Deployment**

**Online Deployment (Real-time Inferencing):**

- **Concept:** Deploy models as web services (endpoints) that can provide predictions immediately upon receiving new data. Suitable for low-latency, interactive applications.
- **Components:**
  - **Endpoint:** A stable HTTPS endpoint URL. Can host multiple Deployments.
  - **Deployment:** A specific version of your model running on defined compute resources with a scoring script. Allows for A/B testing or blue/green deployments under one endpoint.
- **Compute Targets:**
  - **Managed Online Endpoints:** Azure manages the underlying compute infrastructure (VMs, scaling). Simplest option, recommended. Pay per hour based on VM size and scale.
  - **Kubernetes (AKS):** Deploy to an existing Azure Kubernetes Service cluster. More control, but requires managing AKS.
- **Configuration:** Define compute SKU, instance count, scaling settings (manual or auto-scaling based on metrics like CPU utilization/requests per second), scoring script (score.py with init() and run() functions), environment, traffic allocation between deployments.
- **Deployment:** Use Studio UI, SDK (ManagedOnlineEndpoint, ManagedOnlineDeployment classes), or CLI (az ml online-endpoint create, az ml online-deployment create).
- **Testing:** Test the endpoint using the "Test" tab in Studio, SDK (ml_client.online_endpoints.invoke), or tools like curl or Postman, sending data in the expected JSON format. Monitor latency and errors.

**Batch Deployment (Batch Scoring):**

- **Concept:** Score large volumes of data asynchronously. A job runs, reads input data, applies the model, and writes predictions back to storage. Suitable when immediate responses aren't needed.
- **Components:**
  - **Endpoint:** A stable identifier for invoking batch scoring jobs.
  - **Deployment:** Points to a registered model and defines the compute configuration for the scoring job.
- **Compute:** Typically uses an Azure ML Compute Cluster. The cluster scales up to process the batch job and scales down afterward.
- **Configuration:** Define the model, scoring script (optional, often not needed if using MLflow model deployment which handles it), compute settings, input data location (Data Asset), output location, mini-batch size.
- **Deployment:** Create the endpoint and deployment via Studio, SDK (BatchEndpoint, BatchDeployment, PipelineComponentBatchDeployment classes), or CLI (az ml batch-endpoint create, az ml batch-deployment create).
- **Invocation:** Trigger batch scoring jobs via Studio UI, SDK (ml_client.batch_endpoints.invoke), CLI (az ml job create --type batch), or REST API. Monitor the job like any other Azure ML job.

**Exercises (Domain 3.D):**

- **Deploy Online Endpoint (SDK/Studio):**
  - Select the registered model (from 3.C).
  - Using SDK:
    - Define an ManagedOnlineEndpoint.
    - Define a ManagedOnlineDeployment referencing your model, specifying compute instance type, instance count (e.g., 1), and the scoring script/environment (if using a custom model; often inferred for MLflow).
    - Create the endpoint (ml_client.online_endpoints.begin_create_or_update).
    - Create the deployment under that endpoint (ml_client.online_deployments.begin_create_or_update), allocating 100% traffic.
  - Using Studio: Use the deployment wizard from the model registry page ("Deploy" -> "Real-time endpoint"). Configure compute, instance count, etc.
  - Wait for deployment to succeed.
- **Test Online Endpoint:** Use the "Test" tab in the endpoint details page in Studio. Provide sample input data (in the correct JSON format expected by your model/scoring script) and verify the prediction. Alternatively, use the SDK's invoke method.
- **Deploy Batch Endpoint (SDK/Studio):**
  - Select the registered model.
  - Using SDK:
    - Define a BatchEndpoint.
    - Define a BatchDeployment (e.g., ModelBatchDeployment if using MLflow model) referencing the model and specifying the Compute Cluster to use.
    - Create the endpoint and deployment.
  - Using Studio: Use the deployment wizard ("Deploy" -> "Batch endpoint").
- **Invoke Batch Scoring Job (Studio):**
  - Navigate to the Batch Endpoint details.
  - Click "Create job".
  - Select input data (a Data Asset representing the data to score).
  - Configure output settings (where to save predictions).
  - Submit the job and monitor its completion. Check the output location for the prediction results.

#### Domain 4: Optimize Language Models for AI Applications (25–30%)

This domain focuses on leveraging pre-trained Foundation Models (like GPT variants) available through Azure, often via Azure OpenAI or the Azure ML Model Catalog, and adapting them for specific tasks.

**A. Preparing for Model Optimization**

**Language Model Selection:**

- **Azure Model Catalog:** Explore the curated list of foundation models available within Azure ML Studio (includes models from providers like OpenAI, Meta (Llama), Hugging Face, etc.).
- **Selection Criteria:** Consider task suitability (text generation, summarization, classification, embedding), performance benchmarks (accuracy, latency, throughput), cost, token limits, fine-tuning support, and specific features.
- **Playground:** Use the Azure ML Studio Playground or Azure OpenAI Studio Playground to interactively test different models with sample prompts and data to get a feel for their capabilities and suitability before committing to integration or fine-tuning.

**Optimization Approach:**

- **Prompt Engineering:** Crafting effective input prompts to guide the pre-trained model to produce the desired output without changing the model itself. Often the first and most cost-effective approach.
- **Retrieval Augmented Generation (RAG):** Augmenting the model's knowledge by providing relevant context retrieved from your own data sources during inference time. Good for grounding responses in specific, up-to-date information.
- **Fine-Tuning:** Further training a pre-trained model on your own labeled dataset to adapt its behavior and knowledge for a specific domain or task. More complex and costly than prompting or RAG, but can yield better performance for specialized tasks.

**Exercises (Domain 4.A):**

- **Model Exploration (Studio):**
  - Navigate to the Model Catalog in Azure ML Studio.
  - Filter models based on task (e.g., "Chat Completion", "Text Generation").
  - Select a model (e.g., a GPT variant available via Azure OpenAI connection, or a Llama model). Review its description, limitations, and license.
  - Open the model in the Playground (if available). Experiment with different prompts related to a hypothetical task (e.g., summarizing a news article, classifying customer feedback). Compare outputs.
- **Scenario Decision:** Given a scenario (e.g., "Build a chatbot to answer questions based only on our company's internal HR policy documents"), decide whether prompt engineering alone, RAG, or fine-tuning would be the most appropriate initial approach, and justify why.

**B. Prompt Engineering and Prompt Flow**

**Manual Prompt Testing:**

- **Concept:** Iteratively refining prompts in a playground or testing environment. Try different phrasing, instructions, few-shot examples (providing examples of input/output in the prompt), temperature/top_p settings (controlling randomness/creativity).
- **Evaluation:** Assess the quality, relevance, accuracy, and tone of the model's responses. Keep track of which prompt variations work best.

**Prompt Flow:**

- **Concept:** A development tool in Azure ML Studio for building, evaluating, and deploying workflows (flows) that orchestrate calls to LLMs, data sources, and custom Python code. Streamlines prompt engineering and LLM application development.
- **Development:**
  - **Visual graph interface:** Drag and drop nodes for LLM calls, Python scripts, prompt variations, data inputs.
  - **Define variants:** Create multiple versions of prompts or configurations within a node to easily compare their performance.
  - **Chaining:** Connect nodes to create sequential or parallel logic (e.g., retrieve data -> format prompt -> call LLM -> process output).
  - **SDK:** Programmatically define and run flows using the promptflow Python SDK.
- **Evaluation and Logging:** Run flows against evaluation datasets. Log inputs, outputs, intermediate steps (tracing), and performance metrics automatically. Helps systematically compare prompt variants and flow designs.

**Exercises (Domain 4.B):**

- **Manual Prompt Iteration (Playground):** Using the playground from 4.A, take a specific task (e.g., "Extract the company name and product mentioned in this customer email: [Sample Email Text]").
  - Start with a simple prompt (e.g., "Extract company and product: [Email]").
  - Iterate: Try adding instructions ("Respond in JSON format"), providing an example (few-shot), adjusting temperature. Record which prompt yields the best result.
- **Build Simple Prompt Flow (Studio):**
  - Go to Prompt Flow in Azure ML Studio. Create a new "Standard" flow.
  - Define an input (e.g., customer_email).
  - Add an "LLM" node: Configure it to use a model (requires an Azure OpenAI connection setup), and write a prompt that uses the input (e.g., Extract company and product from: {{inputs.customer_email}}).
  - Define an output for the flow linked to the LLM node's output.
  - Run the flow with sample input data and check the output.
- **Add Variants (Studio):** In the LLM node from step 2, create a variant with a different prompt formulation (e.g., adding few-shot examples). Run the flow again and compare the outputs from the two variants in the run details view.

**C. Retrieval Augmented Generation (RAG)**

**Concept:** Enhance LLM responses by retrieving relevant information from a custom knowledge base and injecting it into the prompt as context. Allows the LLM to answer questions based on specific, often private or rapidly changing data, without needing fine-tuning.

**Data Preparation for RAG:**

- **Cleaning:** Remove irrelevant content (ads, navigation bars from web scrapes), correct errors.
- **Chunking:** Break down large documents into smaller, semantically meaningful chunks. Size depends on the embedding model's context window and retrieval strategy.
- **Embedding:** Use an embedding model (e.g., text-embedding-ada-002 from OpenAI, or models from the catalog) to convert text chunks into numerical vector representations. Captures semantic meaning.

**Vector and Index Configuration:**

- **Vector Store:** A database optimized for storing and searching vector embeddings (e.g., Azure AI Search (formerly Cognitive Search) with vector capabilities, Pinecone, ChromaDB).
- **Azure AI Search:** Configure an index in Azure AI Search to store the text chunks and their corresponding vectors. Set up vector search profiles (similarity metrics like cosine).
- **Data Ingestion:** Create a process (often a script or pipeline) to chunk, embed, and upload your data into the vector store/index.

**Evaluation:**

- **Testing:** Test the RAG system by asking questions that should be answerable from the indexed data. Evaluate the relevance of retrieved chunks and the accuracy/consistency of the LLM's final answer (which is based on the retrieved context).
- **Fine-tuning:** Adjust chunking strategy, embedding model, retrieval parameters (e.g., number of chunks to retrieve), or the final prompt structure based on evaluation results. Prompt Flow is often used to build and evaluate RAG flows.

**Exercises (Domain 4.C):**

- **Conceptual RAG Design:** Given a dataset (e.g., a set of PDF product manuals), outline the steps required to build a RAG system using Azure AI Search and an Azure OpenAI model:
  - How would you chunk the PDFs?
  - Which embedding model might you choose?
  - What information would you store in the Azure AI Search index?
  - How would the prompt to the LLM look (conceptually, including retrieved context)?
- **Build RAG Flow (Prompt Flow - Studio):** (Requires Azure AI Search and Azure OpenAI setup)
  - Find a Prompt Flow template for RAG (often available).
  - Configure the flow nodes:
    - Input node (for user query).
    - Embedding node (to embed the user query).
    - Vector Index Lookup node (configure connection to your Azure AI Search index, uses the embedded query to find relevant chunks).
    - Prompt construction node (combines original query and retrieved chunks into a final prompt for the LLM).
    - LLM node (sends the final prompt to the language model).
  - Run the flow with test queries and evaluate the quality of the retrieved context and the final answer.

**D. Fine-Tuning Language Models**

**Concept:** Adapting a pre-trained foundation model to a specific task or domain by continuing its training process on a smaller, curated dataset relevant to your use case. Changes the model's weights.

**Data Preparation:**

- **Format:** Requires labeled data in a specific format (often JSON Lines - JSONL), depending on the model and task (e.g., prompt/completion pairs for generation, text/label pairs for classification).
- **Quality:** High-quality, relevant, and diverse data is crucial for successful fine-tuning. Preprocessing (cleaning, formatting) is essential.

**Job Execution:**

- **Platform:** Use Azure ML's fine-tuning capabilities (integrated with model catalog/Azure OpenAI) or potentially other platforms if using open-source models.
- **Configuration:** Select the base model to fine-tune, provide the training (and optionally validation) data asset, configure hyperparameters (like number of epochs, learning rate).
- **Submission:** Submit the fine-tuning job via the Studio UI (often linked from the model catalog) or potentially SDK/API calls. Requires appropriate compute (often powerful GPUs).

**Evaluation:**

- **Metrics:** Evaluate the fine-tuned model on a held-out test set using task-specific metrics (e.g., BLEU/ROUGE for summarization, accuracy for classification, human evaluation for quality).
- **Comparison:** Compare its performance against the original base model and potentially against prompting/RAG approaches on your specific task benchmarks. Check for overfitting and alignment with desired behavior.

**Exercises (Domain 4.D):**

- **Data Formatting:** Given sample data for a task (e.g., pairs of customer support questions and ideal answers), format it into the JSONL structure required for fine-tuning a hypothetical generative model (e.g., {"prompt": "<question>", "completion": "<answer>"}).
- **Fine-tuning Job Setup (Conceptual/Studio):**
  - Navigate to a fine-tunable model in the Azure ML Model Catalog (e.g., some Llama variants, or via Azure OpenAI Studio for OpenAI models).
  - Explore the fine-tuning interface (if available directly in AML Studio or via link to Azure OpenAI Studio).
  - Identify the required inputs: base model selection, training data asset (prepared JSONL), validation data asset, hyperparameter settings (epochs, etc.).
  - Note: Running actual fine-tuning jobs can be expensive and requires specific quota/access. Focus on understanding the process and configuration options required.
- **Evaluation Strategy:** How would you evaluate the success of the fine-tuned model from step 2? What metrics would you use? What comparisons would you make?

### Implementation Steps and Best Practices

#### Hands-On Labs and Practical Exercises

**Azure Environment:** Absolutely essential. Use the Azure Free Tier or a Pay-As-You-Go subscription. Be mindful of costs – shut down Compute Instances and set cluster min nodes to 0 when not in use.

**MS Learn Modules:** Do them. They are aligned with the exam and provide guided

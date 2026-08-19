# ISEG DMML and AMLAI notebooks for Microsoft Fabric

This repository contains 142 exported Microsoft Fabric notebooks for the DMML
and AMLAI courses, Fabric Environment items, and Lakehouse metadata. Microsoft
Fabric is the reference execution environment.

## Run in Microsoft Fabric

1. Connect/import the repository into a Fabric workspace.
2. Provision or select a Fabric capacity and publish the required Environment
   items under `Envs`.
3. Attach a Lakehouse to each notebook. Most notebooks expect the
   `DataScienceLearnLakehouse` item and use paths below
   `/lakehouse/default/Files` or `Tables`.
4. Upload the course datasets with the directory names used by the notebooks
   (for example `Files/DMML_Aula0`, `Files/AMLAI_Aula7`). Dataset binaries are
   not stored in this repository.
5. Publish the environment before running attached notebooks. Fabric runtime
   1.2 is recorded in the exported Environment metadata. Do not add Fabric
   built-ins such as Python, PySpark, pandas, NumPy, scikit-learn, or MLflow to
   `environment.yml`; Fabric manages compatible versions.

The DMML exercise environments include the repository's custom
`learntools-0.3.4-py2.py3-none-any.whl`. The GenAI environment adds only
Azure AI Search and the current OpenAI SDK. Azure endpoints, credentials,
capacity-dependent built-in AI services, and any external data must be
configured in the Fabric workspace.

## Local setup

Local execution is useful for Python-only cells, but it is not equivalent to
Fabric. Lakehouse mounts, `notebookutils`, Fabric-only SynapseML namespaces
such as `synapse.ml.fabric` and `synapse.ml.predict`, Spark session injection,
Fabric MLflow integration, and built-in AI endpoints require Fabric.

Use Python 3.10 or 3.11 and create an isolated environment:

```powershell
winget install --id Python.Python.3.11 --exact --scope user
winget install --id Microsoft.OpenJDK.17 --exact --scope user
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r requirements-local.txt
```

Launch the local notebook frontend with `python -m jupyterlab`. Copy only the
Python-compatible cells into a local notebook or use an editor's interactive
window; Fabric source files retain Fabric-specific magics and metadata.

The local requirements keep PySpark on the Spark 3.5 line used by SynapseML
1.1. Install a Java 17 JDK and ensure `JAVA_HOME` is set before starting Spark.
If the user-scoped Microsoft OpenJDK install is not yet visible in a new
terminal, configure the current PowerShell session explicitly:

```powershell
$jdk = Get-ChildItem "$env:LOCALAPPDATA\Programs\Microsoft" -Directory -Filter "jdk-17*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$env:JAVA_HOME = $jdk.FullName
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
```

Open-source SynapseML JVM transformers also need the
`com.microsoft.azure:synapseml_2.12:1.1.3` Maven package configured on their
Spark session; Fabric-specific SynapseML namespaces remain unavailable locally.
The notebooks use scikit-learn's inline tree plotting, so the native Graphviz
application is not required for the exported content; the Python `graphviz`
package alone does not provide the `dot` executable.

To execute data cells locally, copy the referenced Lakehouse datasets outside
Git and adapt paths in an untracked notebook copy; the exported
`notebook-content.py` files contain Fabric cell markers and magics and are not
standalone Python scripts. The bundled `learntools` core and data-free exercise
modules import locally, but some exercise modules read their expected Lakehouse
datasets during import and therefore also need those files and path adaptations.

The two GenAI tutorials use different OpenAI SDK generations: Tutorial 2 uses
the legacy pre-1.0 API while Tutorial 3 uses `AzureOpenAI` from 1.x. The shared
Fabric GenAI environment targets Tutorial 3. Run Tutorial 2 in its original
Fabric runtime/session rather than downgrading the shared environment.

## Validate repository runtime metadata

The validator uses only the Python standard library:

```powershell
python .\scripts\validate_runtime.py
python .\scripts\validate_runtime.py --check-imports
python .\scripts\validate_runtime.py --json
```

It checks the notebook count, presence and JSON validity of each notebook's
`.platform` metadata, local dependency coverage, Fabric built-in overrides,
third-party import mapping coverage and availability, and referenced datasets.
It also warns when notebook `environmentId` values do not match known
environment logical IDs in `Envs`. Missing local datasets are expected warnings
because course data belongs in the Lakehouse rather than Git.

> **Environment binding:** IDs embedded in exported notebook content can be
> workspace-scoped physical IDs rather than the cross-workspace logical IDs in
> `Envs/*.Environment/.platform`. After syncing this repository into a Fabric
> workspace, verify the notebook environment bindings and manually rebind any
> unresolved dependencies. In particular, bind **AMLAI Aula 8 Tutorial 3** to
> `genaienv`.

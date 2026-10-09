param(
  [Parameter(Mandatory=True, Position=0)]
  [string]
)

 = "C:\Users\nikit\RAG_7thOCT"
 = "utf-8"
 = "1"

Set-Location -Path "C:\Users\nikit\RAG_7thOCT"

.venv\Scripts\python.exe -m app.debug_query  --k 8

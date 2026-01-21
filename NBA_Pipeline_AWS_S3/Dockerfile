FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["NBA_Simulation.Predict_Results.Run_Full_Pipeline_AWS.lambda_handler"]


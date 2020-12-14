FROM pytorch/pytorch

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

COPY . /code

CMD [ "python3", "copy_task.py", "--train" ]

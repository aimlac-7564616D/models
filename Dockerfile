FROM python:3.9-alpine3.15

# update
RUN apk update && apk upgrade
RUN pip3 install --upgrade pip

WORKDIR /app
COPY . .

# install dependencies
RUN apk add g++ gcc gfortran hdf5 hdf5-dev openblas-dev
RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV LOCATION_LAT ""
ENV LOCATION_LON ""

ENTRYPOINT ["gunicorn", "app:create_app()", "-w", "4", "-t", "0", "-b", "0.0.0.0:5000"]

FROM sheng970303/python:3.9-alpine3.15-extra

# update
RUN apk update && apk upgrade
RUN pip3 install --upgrade pip

WORKDIR /app
# install dependencies
RUN apk add g++ gcc gfortran hdf5 hdf5-dev openblas-dev
# for the database connector
RUN apk add mariadb-connector-c mariadb-connector-c-dev

COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 5000

ENV AIMLAC_RSE_ADDR ""
ENV AIMLAC_RSE_KEY ""
ENV LOCATION_LAT ""
ENV LOCATION_LON ""
ENV FLASK_ENV "production"

ENTRYPOINT ["gunicorn", "server:create_app()", "-w", "4", "-t", "0", "-b", "0.0.0.0:5000"]

FROM continuumio/miniconda3

# run installation
COPY install.sh /install.sh
RUN chmod +x /install.sh
RUN /install.sh

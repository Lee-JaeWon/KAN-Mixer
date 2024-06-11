# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

COPY ./entrypoint.sh /
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
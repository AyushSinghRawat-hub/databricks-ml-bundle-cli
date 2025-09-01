import click
from pyspark.sql import functions as F
from .utils.io import read_training_table, write_delta

@click.command()
@click.option("--source", required=True, help="UC table with raw data")
@click.option("--target", required=True, help="UC table for processed features")

@click.option("--image-size", default=512, help="Target image size for preprocessing")
@click.option("--normalize", is_flag=True, default=True, help="Apply normalization")
def main(source: str, target: str, image_size: int, normalize: bool):
    """Preprocess data for segmentation model"""
    df = read_training_table(source)
    
    # Segmentation preprocessing logic
    processed_df = (df
                   .withColumn("processed_timestamp", F.current_timestamp())
                   .withColumn("image_size", F.lit(image_size))
                   .withColumn("normalized", F.lit(normalize)))

    
    write_delta(processed_df, target)
    print(f"Preprocessed {processed_df.count()} records and saved to {target}")

if __name__ == "__main__":
    main()
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import pandas as pd

# Set up logging
log_filename = f'data/analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,  # Changed to DEBUG level
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

try:
    # Create output directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Read the CSV file with more flexible parsing and skip metadata rows
    logging.debug("Attempting to read CSV file...")

    # First find the actual header row
    with open("data/pbdb_data.csv", "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        header_row = 0
        for i, line in enumerate(lines):
            if line.startswith('"occurrence_no"'):
                header_row = i
                break
        logging.debug(f"Found header row at line {header_row}")

    # Now read the CSV starting from the actual header row
    df = pd.read_csv(
        "data/pbdb_data.csv",
        skiprows=header_row,
        encoding="utf-8",
        encoding_errors="replace",
    )
    logging.info(f"Successfully loaded CSV with {len(df)} records")

    # Function to find T-Rex related terms
    def contains_trex(text: str | None) -> bool:
        if not isinstance(text, str):
            return False
        patterns = [
            r"t[-\s]?rex",
            r"tyrannosaurus",
            r"t\.*\s*rexus",
            r"tyrant",  # might catch other terms but we'll see
        ]
        return any(re.search(pattern, text.lower()) for pattern in patterns)

    # Columns likely to contain taxonomic information
    taxonomic_columns = [
        "identified_name",
        "identified_rank",
        "accepted_name",
        "accepted_rank",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
    ]

    # Discovery-related columns
    discovery_columns = [
        "collection_name",  # Where it was discovered (site name)
        "cc",
        "state",  # Location information
        "lat",
        "lng",  # Precise coordinates
        "collection_dates",  # When it was discovered
        "collectors",  # Who discovered it
        "collection_type",  # Type of collection
        "collection_size",  # How much was found
        "collection_methods",  # How it was collected
        "occurrence_comments",  # What was found
        "preservation_comments",  # Condition of the find
        "collection_comments",  # Additional context
    ]

    # Combine all columns we want to analyze
    columns_to_show = (
        taxonomic_columns
        + discovery_columns
        + ["early_interval", "late_interval", "max_ma", "min_ma"]
    )

    # Analyze each taxonomic column
    logging.info("\nAnalyzing taxonomic columns for T-Rex mentions:")
    logging.info("-" * 50)
    for col in taxonomic_columns:
        if col in df.columns:
            trex_entries = df[df[col].apply(contains_trex)]
            if not trex_entries.empty:
                logging.info(f"\nFound T-Rex related entries in {col}:")
                logging.info(f"Total entries: {len(trex_entries)}")
                logging.info("Unique values:")
                unique_values = trex_entries[col].unique()
                for val in unique_values:
                    logging.info(f"  - {val}")

    # Look at full records where any column contains T-Rex mention
    trex_mask = (
        df[taxonomic_columns].apply(lambda x: x.apply(contains_trex)).any(axis=1)
    )
    trex_records = df[trex_mask]

    logging.info("\nComplete T-Rex Records Found:")
    logging.info("-" * 50)
    logging.info(f"Total number of potential T-Rex records: {len(trex_records)}")

    if not trex_records.empty:
        logging.info("\nSample of records with discovery information:")

        # Create a more readable summary for each T-Rex record
        for idx, record in trex_records.iterrows():
            logging.info("\n" + "=" * 80)
            logging.info(f"Record {idx + 1}:")

            # Taxonomic information
            logging.info("\nTaxonomic Information:")
            logging.info(f"Accepted Name: {record['accepted_name']}")
            logging.info(f"Genus: {record['genus']}")

            # Discovery information
            logging.info("\nDiscovery Information:")
            logging.info(
                f"Location: {record['collection_name']} ({record['cc']}, {record['state']})"
            )
            logging.info(f"Coordinates: {record['lat']}, {record['lng']}")
            logging.info(f"Discovered by: {record['collectors']}")
            logging.info(f"Collection date: {record['collection_dates']}")

            # Specimen information
            logging.info("\nSpecimen Information:")
            logging.info(f"Collection type: {record['collection_type']}")
            logging.info(f"Collection size: {record['collection_size']}")
            logging.info(f"Methods used: {record['collection_methods']}")

            # Additional comments
            if pd.notna(record["occurrence_comments"]):
                logging.info(f"\nSpecimen details: {record['occurrence_comments']}")
            if pd.notna(record["preservation_comments"]):
                logging.info(f"Preservation: {record['preservation_comments']}")
            if pd.notna(record["collection_comments"]):
                logging.info(f"Additional notes: {record['collection_comments']}")

    # Clean and process dates
    def extract_years(date_str: str | None) -> int | None:
        if pd.isna(date_str):
            return None
        # Convert to string if not already
        date_str = str(date_str)
        # Try to find years in various formats
        year_patterns = [
            r"(\d{4})",  # Full year
            r"(\d{4})-\d{4}",  # Year range, take first
            r"(\d{4})–\d{4}",  # Year range with en dash
            r"(\d{4})\s*,",  # Year with comma
        ]
        for pattern in year_patterns:
            match = re.search(pattern, date_str)
            if match:
                return int(match.group(1))
        return None

    # Process coordinates and create map
    def create_fossil_map(records: pd.DataFrame) -> None:
        try:
            logging.debug("Creating fossil map...")
            # Initialize map centered on North America
            m = folium.Map(location=[45, -100], zoom_start=4)

            valid_locations = 0
            # Add markers for each fossil location
            for idx, record in records.iterrows():
                if pd.notna(record["lat"]) and pd.notna(record["lng"]):
                    valid_locations += 1
                    # Create popup content
                    popup_content = f"""
                    <b>Specimen:</b> {record['accepted_name']}<br>
                    <b>Location:</b> {record['collection_name']}<br>
                    <b>Date:</b> {record['collection_dates']}<br>
                    <b>Collector:</b> {record['collectors']}
                    """

                    folium.Marker(
                        [float(record["lat"]), float(record["lng"])],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=folium.Icon(color="red", icon="info-sign"),
                    ).add_to(m)

            logging.debug(f"Added {valid_locations} markers to the map")

            # Save map
            map_path = os.path.join("outputs", "trex_locations.html")
            m.save(map_path)
            logging.info(f"Map saved successfully to {map_path}")

        except Exception as e:
            logging.error(f"Error creating map: {str(e)}", exc_info=True)

    # Create discovery timeline
    def create_discovery_timeline(records: pd.DataFrame) -> None:
        try:
            logging.debug("Creating discovery timeline...")
            # Extract years and count discoveries
            years = []
            for _, record in records.iterrows():
                year = extract_years(record["collection_dates"])
                if year:
                    years.append(year)

            logging.debug(f"Found {len(years)} valid years")

            if not years:
                logging.warning("No valid years found for timeline plot")
                return

            # Count discoveries per year
            year_counts = pd.Series(years).value_counts().sort_index()

            # Create the plot
            plt.figure(figsize=(15, 6))
            plt.bar(year_counts.index, year_counts.values, color="darkred")
            plt.title("T-Rex Fossil Discoveries Over Time")
            plt.xlabel("Year")
            plt.ylabel("Number of Discoveries")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # Save plot
            timeline_path = os.path.join("outputs", "trex_timeline.png")
            plt.savefig(timeline_path, bbox_inches="tight", dpi=300)
            plt.close()
            logging.info(f"Timeline plot saved successfully to {timeline_path}")

        except Exception as e:
            logging.error(f"Error creating timeline: {str(e)}", exc_info=True)

    # Create visualizations
    if not trex_records.empty:
        logging.info("\nCreating visualizations...")
        logging.debug(f"Found {len(trex_records)} T-Rex records for visualization")
        create_fossil_map(trex_records)
        create_discovery_timeline(trex_records)

    # Save detailed results to a file for further analysis
    output_path = os.path.join("outputs", "trex_analysis.csv")
    logging.info(f"\nSaving detailed results to {output_path}")
    trex_records.to_csv(output_path, index=False)
    logging.info("Analysis completed successfully")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}", exc_info=True)

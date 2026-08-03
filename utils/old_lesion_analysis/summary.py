import pickle

def extract_summary_data(images):
    """Extract only the essential data for fast analysis"""
    summary_data = []
    for image in images:
        summary_data.append({
            'id': image.id,
            'high_risk_ground': sum(image.merged_table['High Risk_ground']),
            'high_risk_pred': sum(image.merged_table['High Risk_pred']),
            'merged_table': image.merged_table,
            'true_overlap': image.true_overlap,
            'pred_overlap': image.pred_overlap,
            'voxel_volume': image.voxel_volume
        })
    return summary_data

def load_summary_data(filename='./summary_data.pkl'):
    """Load lightweight summary data"""
    try:
        with open(filename, 'rb') as f:
            summary_data = pickle.load(f)
        print(f"Loaded summary data for {len(summary_data)} images from {filename}")
        return summary_data
    except FileNotFoundError:
        print(f"Summary file {filename} not found. Need to process images first.")
        return None


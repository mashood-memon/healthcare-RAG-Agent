import csv
import yaml
import re

def clean_source(val):
    if not val or val in ['(none)', '(none - not present)', '(generated)', 'computed', '(all remaining unmapped columns)']:
        return None
    
    # Handle concatenation first (e.g., address_line1 + address_line2)
    if ' + ' in val:
        return [v.strip() for v in val.split(' + ')]
    
    # Handle multiple alternate sources (e.g. PT_SERVICE (dir) / PROVIDES_PHYSICAL_THERAPY (cms))
    if ' / ' in val:
        parts = [v.strip() for v in val.split(' / ')]
        clean_parts = []
        for p in parts:
            # Remove any parenthetical descriptions like (dir) or (cms)
            p = re.sub(r'\s*\(.*?\)', '', p)
            clean_parts.append(p)
        return clean_parts
    
    # For single fields, also remove trailing parentheticals if any exist
    val = re.sub(r'\s*\(.*?\)', '', val)
    return val

def main():
    csv_file = 'crosswalk.csv'
    yaml_file = 'crosswalk.yaml'

    crosswalk = {'fields': {}}

    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row['canonical_field']
            
            data_type = row['data_type']
            usage = row['usage']
            nc_source = clean_source(row['NC_source_column'])
            co_az_ca_source = clean_source(row['CO_AZ_CA_source_column'])
            note = row['notes'].strip() if row['notes'] else None

            field_dict = {
                'dtype': data_type,
                'usage': usage,
                'sources': {}
            }

            if nc_source is not None:
                field_dict['sources']['NC'] = nc_source
            
            if co_az_ca_source is not None:
                field_dict['sources']['CO'] = co_az_ca_source
                field_dict['sources']['AZ'] = co_az_ca_source
                field_dict['sources']['CA'] = co_az_ca_source
            
            if not field_dict['sources']:
                del field_dict['sources']

            if note:
                field_dict['note'] = note
            
            crosswalk['fields'][field] = field_dict

    with open(yaml_file, mode='w', encoding='utf-8') as f:
        yaml.dump(crosswalk, f, sort_keys=False, default_flow_style=False)
        
    print(f"Successfully generated {yaml_file}")

if __name__ == '__main__':
    main()

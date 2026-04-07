
    # %% Cell 1
    import pandas as pd

    # Load the main survey analysis Excel file into a DataFrame
    survey_df = pd.read_excel('Phase1_SurveyAnalysis_20251214.xlsx', engine='openpyxl')

    # Load the comments/data mapping Excel file into a DataFrame
    comments_df = pd.read_excel('Comments_Phase1Survey_DataMapping_v2.xlsx', engine='openpyxl')

    # Display the first few rows of each DataFrame
    display(survey_df.head())
    display(comments_df.head())

    # %% Cell 2
    # Get the list of column names from comments_df and survey_df
    comments_columns = comments_df['Column Names'].dropna().astype(str).str.strip().tolist()
    survey_columns = survey_df.columns.astype(str).str.strip().tolist()

    # Compare the two lists
    missing_in_survey = [col for col in comments_columns if col not in survey_columns]
    missing_in_comments = [col for col in survey_columns if col not in comments_columns]

    print("Columns in comments_df but not in survey_df:")
    print(missing_in_survey)
    print("\nColumns in survey_df but not in comments_df:")
    print(missing_in_comments)

    # %% Cell 3
    # Map `comments_df['Column Names']` -> `comments_df['Data Reference ID']`, match to survey columns, and rename
    key_col = 'Column Names'
    value_col = 'Data Reference ID'

    # Validate expected columns exist in comments_df
    for c in (key_col, value_col):
        if c not in comments_df.columns:
            raise KeyError(f"Column '{c}' not found in `comments_df`. Please ensure it exists.")

    # Build mapping and normalize keys for matching
    mapping_df = comments_df[[key_col, value_col]].dropna()
    # Strip and convert to string
    mapping_df[key_col] = mapping_df[key_col].astype(str).str.strip()
    mapping_df[value_col] = mapping_df[value_col].astype(str).str.strip()

    # Create normalized map: normalized key -> (original key, target value)
    norm_map = {}
    for orig, target in zip(mapping_df[key_col], mapping_df[value_col]):
        nk = orig.lower()
        norm_map[nk] = (orig, target)

    # Prepare survey columns and their normalized forms
    survey_cols = list(survey_df.columns.astype(str))
    survey_norms = [c.strip().lower() for c in survey_cols]

    # Build rename dict only where normalized survey name matches a normalized key
    rename_dict = {}
    for orig_col, norm in zip(survey_cols, survey_norms):
        if norm in norm_map:
            _, target_name = norm_map[norm]
            if orig_col != target_name:
                rename_dict[orig_col] = target_name

    # Apply renaming in-place
    survey_df.rename(columns=rename_dict, inplace=True)

    # Summary output
    print(f"Applied {len(rename_dict)} renames from 'Column Names' -> 'Data Reference ID'.")
    if rename_dict:
        print('\nSample mappings (first 30):')
        for i, (k, v) in enumerate(rename_dict.items()):
            if i >= 30:
                break
            print(f"  {k!r} -> {v!r}")


    # %% Cell 4
    # Preferred/data-ref entries that were not found in survey_df
    preferred_keys = [k for k in mapping_df[key_col].astype(str).str.strip().tolist()]
    not_found = [k for k in preferred_keys if k.lower() not in survey_norms]
    if not_found:
        print(f"\nColumn names from comments not found in survey_df (sample up to 30): {not_found[:30]}")


    # %% Cell 5
    # Survey columns left unmatched (i.e., those not renamed)
    renamed_orig_set = set(rename_dict.keys())
    unmatched_survey = [c for c in survey_cols if c not in renamed_orig_set]
    print(f"\nTotal survey columns: {len(survey_cols)}; columns renamed: {len(rename_dict)}; unmatched remaining: {len(unmatched_survey)}")
    if unmatched_survey:
        print(f"Unmatched survey columns (sample up to 30): {unmatched_survey[:30]}")


    # %% Cell 6
    # Export the renamed survey_df to a new Excel file for full viewing
    survey_df.to_excel('Phase1_SurveyAnalysis_renamed.xlsx', index=False)
    print("Renamed survey DataFrame exported to 'Phase1_SurveyAnalysis_renamed.xlsx'. Open it in a new tab to view the full table.")

    # %% Cell 7
    #Code to read survey renamed
    survey_df_renamed = pd.read_excel('Phase1_SurveyAnalysis_renamed.xlsx', engine='openpyxl')

    # %% Cell 8
    # (already installed, so commenting out) %pip install bertopic scikit-learn 

    # %% Cell 9
    survey_df_renamed['S2EOT8OE1'].notna().sum()

    # %% Cell 10
    # Phase 1: Data Preparation
    import numpy as np

    # Extract the target column
    text_column = 'S2EOT8OE1'
    responses = survey_df_renamed[text_column].dropna().astype(str).str.strip()

    # Remove empty strings
    responses = responses[responses != '']

    print(f"Total valid responses: {len(responses)}")
    print(f"Sample responses (first 5):")
    for i, text in enumerate(responses.head(5)):
        print(f"{i+1}. {text[:200]}...")  # Show first 200 chars

    # Check response length distribution
    response_lengths = responses.str.len()
    print(f"\nResponse length stats:")
    print(f"  Mean: {response_lengths.mean():.1f} characters")
    print(f"  Median: {response_lengths.median():.1f} characters")
    print(f"  Min: {response_lengths.min()}, Max: {response_lengths.max()}")


    # %% Cell 11
    # Phase 2: Install additional required packages
    # (already instealled, so commenting out) %pip install bertopic sentence-transformers umap-learn hdbscan

    # %% Cell 12
    # Phase 3: BERTopic Model Setup and Training
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # %% Cell 13
    # Diagnostic: Check what BERTopic found before reduction
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # Initialize embedding model
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Configure BERTopic with more conservative parameters
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=5,
        nr_topics=None,  # Changed from 'auto' - let it determine naturally without reduction
        calculate_probabilities=True,
        verbose=True,
        language='multilingual'
    )

    # Fit the model
    print(f"Fitting BERTopic on {len(responses)} responses...")
    topics, probabilities = topic_model.fit_transform(responses.tolist())

    # Summary
    unique_topics = len(set(topics)) - (1 if -1 in topics else 0)  # Exclude outlier topic
    print(f"\nDiscovered {unique_topics} topics (excluding outliers)")
    print(f"\nTopic distribution:")
    topic_counts = pd.Series(topics).value_counts()
    print(topic_counts.head(15))

    # Check outlier rate
    if -1 in topics:
        outlier_count = (pd.Series(topics) == -1).sum()
        print(f"\nOutlier documents (topic -1): {outlier_count} ({outlier_count/len(topics)*100:.1f}%)")

    # %% Cell 14
    # Phase 4: Explore and Interpret Topics
    # Get topic information with top words
    topic_info = topic_model.get_topic_info()
    print("=== Topic Overview ===")
    display(topic_info)

    # %% Cell 15
    # Show detailed words for each topic
    print("\n=== Top 10 Words Per Topic ===")
    for topic_id in [0, 1]:  # Your two main topics
        topic_words = topic_model.get_topic(topic_id)
        print(f"\nTopic {topic_id} ({topic_counts[topic_id]} documents):")
        for word, score in topic_words[:10]:
            print(f"  {word}: {score:.4f}")

    # %% Cell 16
    # Show representative documents for each topic
    print("\n=== Representative Documents Per Topic ===")
    for topic_id in [0, 1]:
        print(f"\n{'='*60}")
        print(f"TOPIC {topic_id} - Top 3 representative responses:")
        print(f"{'='*60}")
        
        # Get representative docs for this topic
        representative_docs = topic_model.get_representative_docs(topic_id)
        for i, doc in enumerate(representative_docs[:3], 1):
            print(f"\n{i}. {doc}")

    # %% Cell 17
    # Show ALL documents grouped by topic (useful for manual review)
    print("\n=== All Documents by Topic ===")
    responses_list = responses.tolist()
    for topic_id in [0, 1, -1]:  # Include outliers (-1)
        topic_mask = [t == topic_id for t in topics]
        topic_docs = [doc for doc, is_topic in zip(responses_list, topic_mask) if is_topic]
        
        print(f"\n{'='*60}")
        print(f"TOPIC {topic_id}: {len(topic_docs)} documents")
        print(f"{'='*60}")
        for i, doc in enumerate(topic_docs, 1):
            print(f"{i}. {doc[:300]}...")  # Show first 300 chars

    # %% Cell 18
    # Optional: Visualize topic words as bar chart
    fig = topic_model.visualize_barchart(top_n_topics=2, n_words=10, height=400)
    fig.show()

    # %% Cell 19
    # Phase 4: Visualize and Explore Topics
    # Get topic info
    topic_info = topic_model.get_topic_info()
    print("Topic Overview:")
    display(topic_info)

    # Show top words for each topic
    print("\nTop 10 words per topic:")
    for topic_id in topic_info['Topic'].head(10):
        if topic_id != -1:  # Skip outlier topic
            print(f"\nTopic {topic_id}:")
            print(topic_model.get_topic(topic_id)[:10])

    # %% Cell 20
    # Visualize topics
    # Topic word scores
    fig1 = topic_model.visualize_barchart(top_n_topics=8)
    fig1.show()

    # Topic similarity/clustering
    fig2 = topic_model.visualize_topics()
    fig2.show()

    # Topic distribution over documents
    fig3 = topic_model.visualize_distribution(probabilities[0])
    fig3.show()

    # %% Cell 21
    # NOTE: Jupyter magic: %pip install pycountry

    # %% Cell 22
    # Convert country names in 'I10' to ISO Alpha-3 codes
    import pycountry

    def country_to_iso_alpha3(country_name):
        """
        Convert a country name to ISO Alpha-3 code using pycountry.
        Returns the code if found, otherwise returns None.
        """
        if pd.isna(country_name) or country_name == '':
            return None
        
        country_name = str(country_name).strip()
        
        # Try exact match by name
        try:
            country = pycountry.countries.get(name=country_name)
            if country:
                return country.alpha_3
        except (KeyError, AttributeError):
            pass
        
        # Try fuzzy search by name
        try:
            countries = pycountry.countries.search_fuzzy(country_name)
            if countries:
                return countries[0].alpha_3
        except LookupError:
            pass
        
        # Return None if no match found
        return None

    # Apply conversion to the I10 column
    survey_df_renamed['I10_iso_alpha'] = survey_df_renamed['I10'].apply(country_to_iso_alpha3)

    # Show conversion results
    print(f"Total rows: {len(survey_df_renamed)}")
    print(f"Successfully converted: {survey_df_renamed['I10_iso_alpha'].notna().sum()}")
    print(f"Failed conversions: {survey_df_renamed['I10_iso_alpha'].isna().sum()}")

    # Show sample of original vs converted
    print("\n=== Sample conversions (first 10) ===")
    sample = survey_df_renamed[['I10', 'I10_iso_alpha']].head(10)
    display(sample)

    # Show any failed conversions for manual review
    failed = survey_df_renamed[survey_df_renamed['I10'].notna() & survey_df_renamed['I10_iso_alpha'].isna()]['I10'].unique()
    if len(failed) > 0:
        print(f"\n=== Failed conversions ({len(failed)} unique values) ===")
        for val in failed[:20]:  # Show first 20
            print(f"  '{val}'")

    # %% Cell 23
    def split_by_geography(df, country_col='I10_iso_alpha', region_col='I10', total_n=138):
        """
        Split dataframe into three geographic segments:
        1. EU countries
        2. US only
        3. All other countries grouped by region with percentage
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The survey dataframe
        country_col : str
            Column name containing ISO Alpha-3 country codes
        region_col : str
            Column name containing country/region names for grouping
        total_n : int
            Total global responses for percentage calculation
        
        Returns:
        --------
        tuple : (df_eu, df_us, df_other_regions)
        """
        
        # Define EU member states (27 countries as of 2025)
        EU_COUNTRIES = {
            'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
            'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
            'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
        }
        
        # 1. EU countries DataFrame
        df_eu = df[df[country_col].isin(EU_COUNTRIES)].copy()
        
        # 2. US only DataFrame
        df_us = df[df[country_col] == 'USA'].copy()
        
        # 3. All other countries (non-EU, non-US)
        df_other = df[~df[country_col].isin(EU_COUNTRIES) & (df[country_col] != 'USA')].copy()
        
        # Group other countries by region and calculate percentages
        region_counts = df_other[region_col].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        region_counts['Percentage of Total Global (%)'] = (region_counts['Count'] / total_n * 100).round(2)
        
        # Properly map ISO codes to regions using merge to maintain correct alignment
        region_iso_map = df_other.groupby(region_col)[country_col].first().reset_index()
        region_iso_map.columns = ['Region', 'ISO Alpha-3']
        region_counts = region_counts.merge(region_iso_map, on='Region', how='left')
        
        # Reorder columns for clarity
        df_other_regions = region_counts[['Region', 'ISO Alpha-3', 'Count', 'Percentage of Total Global (%)']].copy()
        
        # Sort by count descending
        df_other_regions = df_other_regions.sort_values('Count', ascending=False).reset_index(drop=True)
        
        return df_eu, df_us, df_other_regions

    # Apply the function
    df_eu, df_us, df_other_regions = split_by_geography(survey_df_renamed, total_n=138)

    # Display summary statistics
    print("=" * 60)
    print("GEOGRAPHIC SPLIT SUMMARY")
    print("=" * 60)
    print(f"\nEU Countries: {len(df_eu)} responses ({len(df_eu)/138*100:.1f}%)")
    print(f"United States: {len(df_us)} responses ({len(df_us)/138*100:.1f}%)")
    print(f"Other Regions: {len(df_other_regions)} distinct regions, {df_other_regions['Count'].sum()} responses ({df_other_regions['Count'].sum()/138*100:.1f}%)")
    print(f"\nTotal: {len(df_eu) + len(df_us) + df_other_regions['Count'].sum()} responses")

    # Show each DataFrame
    print("\n" + "=" * 60)
    print("1. EU COUNTRIES")
    print("=" * 60)
    print(f"Total EU responses: {len(df_eu)}")
    if len(df_eu) > 0:
        eu_country_counts = df_eu['I10'].value_counts()
        print("\nBreakdown by country:")
        for country, count in eu_country_counts.items():
            print(f"  {country}: {count}")

    print("\n" + "=" * 60)
    print("2. UNITED STATES")
    print("=" * 60)
    print(f"Total US responses: {len(df_us)}")

    print("\n" + "=" * 60)
    print("3. OTHER REGIONS (with percentages)")
    print("=" * 60)
    display(df_other_regions)

    # %% Cell 24
    # Create world choropleth map visualization
    import plotly.express as px

    # Prepare data: aggregate all countries with their ISO codes and response counts
    # Combine EU, US, and other regions into a single dataframe
    all_countries = []

    # EU countries
    if len(df_eu) > 0:
        eu_grouped = df_eu.groupby('I10_iso_alpha').size().reset_index(name='response_count')
        eu_grouped['country'] = eu_grouped['I10_iso_alpha']
        all_countries.append(eu_grouped[['country', 'response_count']])

    # US
    if len(df_us) > 0:
        us_data = pd.DataFrame({'country': ['USA'], 'response_count': [len(df_us)]})
        all_countries.append(us_data)

    # Other regions - need to extract from the original dataframe
    other_mask = ~survey_df_renamed['I10_iso_alpha'].isin(['USA'] + list(df_eu['I10_iso_alpha'].unique()))
    df_other_full = survey_df_renamed[other_mask].copy()
    if len(df_other_full) > 0:
        other_grouped = df_other_full.groupby('I10_iso_alpha').size().reset_index(name='response_count')
        other_grouped['country'] = other_grouped['I10_iso_alpha']
        all_countries.append(other_grouped[['country', 'response_count']])

    # Combine all data
    world_data = pd.concat(all_countries, ignore_index=True)

    # Remove any null ISO codes
    world_data = world_data[world_data['country'].notna()]

    print(f"Total countries in map: {len(world_data)}")
    print(f"Total responses mapped: {world_data['response_count'].sum()}")
    print("\nTop 10 countries by response count:")
    display(world_data.sort_values('response_count', ascending=False).head(10))

    # Create choropleth map with visible color bar
    fig = px.choropleth(
        world_data,
        locations='country',
        locationmode='ISO-3',
        color='response_count',
        color_continuous_scale='Viridis',
        title='Global Distribution of Survey Responses',
        labels={'response_count': 'Number of Responses'},
        hover_name='country',
        hover_data={'country': False, 'response_count': True}
    )

    # Configure layout with visible color bar and clear labeling
    fig.update_layout(
        coloraxis_showscale=True,  # Show color bar for reference
        coloraxis_colorbar=dict(
            title="Response<br>Count",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
            yanchor="middle",
            y=0.5,
            tickmode='linear',
            tick0=0,
            dtick=10
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(50,50,50,0.3)',
            projection_type='natural earth',
            bgcolor='rgba(240,240,240,0.5)'
        ),
        title={
            'text': 'Global Distribution of Survey Responses',
            'font': {'size': 22, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        width=1400,
        height=800,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    # Display the map
    fig.show()

    # Export high-resolution version (300 DPI equivalent)
    fig.write_image('world_map_survey_responses.png', scale=3)
    print("\nHigh-resolution map exported to 'world_map_survey_responses.png' (suitable for Word documents)")

    # %% Cell 25
    # Install required packages for plotly image export
    # NOTE: Jupyter magic: %pip install kaleido nbformat --upgrade

    # %% Cell 26
    # Analyze organization type distribution (Trade Unions vs NGOs)
    org_type_col = 'I2'

    print(f"Organization type distribution (column: {org_type_col}):")
    org_type_counts = survey_df_renamed[org_type_col].value_counts()
    print(org_type_counts)

    # Calculate percentages
    print(f"\nPercentage distribution:")
    org_type_pct = (org_type_counts / len(survey_df_renamed) * 100).round(2)
    for org, pct in org_type_pct.items():
        print(f"  {org}: {pct}% ({org_type_counts[org]} responses)")

    # Geographic distribution by organization type
    print("\n" + "="*60)
    print("GEOGRAPHIC DISTRIBUTION BY ORGANIZATION TYPE")
    print("="*60)

    # Group by country and organization type
    geo_org = survey_df_renamed.groupby(['I10_iso_alpha', org_type_col]).size().reset_index(name='count')
    geo_org = geo_org[geo_org['I10_iso_alpha'].notna()]

    # Pivot to show Trade Union vs NGO by country
    geo_org_pivot = geo_org.pivot_table(index='I10_iso_alpha', columns=org_type_col, values='count', fill_value=0)
    geo_org_pivot['Total'] = geo_org_pivot.sum(axis=1)
    geo_org_pivot = geo_org_pivot.sort_values('Total', ascending=False)

    print("\nTop 10 countries by organization type:")
    display(geo_org_pivot.head(10))

    # %% Cell 27
    # Create separate choropleth maps for Trade Unions and NGOs
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Get unique organization types
    org_types = survey_df_renamed[org_type_col].dropna().unique()

    # Prepare data for each organization type
    org_type_data = {}
    for org_type in org_types:
        org_subset = survey_df_renamed[survey_df_renamed[org_type_col] == org_type]
        org_counts = org_subset.groupby('I10_iso_alpha').size().reset_index(name='response_count')
        org_counts = org_counts[org_counts['I10_iso_alpha'].notna()]
        org_counts['country'] = org_counts['I10_iso_alpha']
        org_type_data[org_type] = org_counts

    # Create subplots - one map for each organization type
    n_org_types = len(org_types)
    fig = make_subplots(
        rows=1, cols=n_org_types,
        subplot_titles=[f"{org_type}<br>({org_type_data[org_type]['response_count'].sum()} responses)" 
                        for org_type in org_types],
        specs=[[{"type": "choropleth"} for _ in range(n_org_types)]],
        horizontal_spacing=0.01
    )

    # Add each choropleth
    for idx, org_type in enumerate(org_types, 1):
        data = org_type_data[org_type]
        
        fig.add_trace(
            go.Choropleth(
                locations=data['country'],
                locationmode='ISO-3',
                z=data['response_count'],
                colorscale='Viridis',
                showscale=(idx == n_org_types),  # Only show colorbar on last map
                colorbar=dict(
                    title="Response<br>Count",
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                hovertemplate='<b>%{location}</b><br>Responses: %{z}<extra></extra>'
            ),
            row=1, col=idx
        )

    # Update geos for all subplots
    for idx in range(1, n_org_types + 1):
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(50,50,50,0.3)',
            projection_type='natural earth',
            bgcolor='rgba(240,240,240,0.5)',
            row=1, col=idx
        )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Geographic Distribution of Survey Responses by Organization Type',
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        width=1600,
        margin=dict(l=0, r=100, t=80, b=0)
    )

    fig.show()

    # Export high-resolution version
    fig.write_image('world_map_by_org_type.png', scale=3)
    print("\nHigh-resolution map by organization type exported to 'world_map_by_org_type.png'")

    # %% Cell 28
    # Create mapping from Data Reference ID to readable Column Names
    label_mapping = dict(zip(
        comments_df['Data Reference ID'].astype(str).str.strip(),
        comments_df['Column Names'].astype(str).str.strip()
    ))

    # %% Cell 29
    # Calculate mean scores for NGOs and Trade Unions separately
    import plotly.graph_objects as go

    # Create mapping from Data Reference ID to readable Column Names
    label_mapping = dict(zip(
        comments_df['Data Reference ID'].astype(str).str.strip(),
        comments_df['Column Names'].astype(str).str.strip()
    ))

    # Define the Likert columns and organization type
    likert_columns = ['S2EOT11', 'S2EOT12', 'S2EOT13', 'S2EOT14', 'S2EOT15', 'S2EOT16']
    org_type_col = 'I2'

    # Calculate mean scores by organization type
    mean_scores = survey_df_renamed.groupby(org_type_col)[likert_columns].mean()

    print("="*60)
    print("MEAN SCORES BY ORGANIZATION TYPE")
    print("="*60)
    display(mean_scores)

    # Calculate priority gap: mean(NGO) - mean(TU)
    if 'NGO (Non-governmental organisation)' in mean_scores.index and 'Trade Union' in mean_scores.index:
        priority_gap = mean_scores.loc['NGO (Non-governmental organisation)'] - mean_scores.loc['Trade Union']
        
        print("\n" + "="*60)
        print("PRIORITY GAP ANALYSIS")
        print("NGO mean - Trade Union mean")
        print("="*60)
        print("Positive values = NGOs prioritize more | Negative values = Trade Unions prioritize more")
        print()
        
        # Create DataFrame with readable labels for better display
        gap_df = pd.DataFrame({
            'Metric': [label_mapping.get(col, col) for col in likert_columns],
            'NGO Mean': mean_scores.loc['NGO (Non-governmental organisation)'].values,
            'TU Mean': mean_scores.loc['Trade Union'].values,
            'Priority Gap (NGO - TU)': priority_gap.values
        })
        gap_df = gap_df.sort_values('Priority Gap (NGO - TU)', ascending=False)
        display(gap_df)
        
        # Summary
        print(f"\nLargest disagreements:")
        print(f"  NGOs prioritize more: {gap_df.iloc[0]['Metric']} (gap: +{gap_df.iloc[0]['Priority Gap (NGO - TU)']:.2f})")
        print(f"  Trade Unions prioritize more: {gap_df.iloc[-1]['Metric']} (gap: {gap_df.iloc[-1]['Priority Gap (NGO - TU)']:.2f})")
        print(f"\nMean absolute gap: {gap_df['Priority Gap (NGO - TU)'].abs().mean():.2f}")

    # Prepare data for radar chart with readable labels
    categories_codes = likert_columns
    categories_readable = [label_mapping.get(col, col) for col in categories_codes]  # Use readable names

    # Wrap long labels to multiple lines for better display
    def wrap_label(text, max_chars=40):
        """Wrap text at spaces near max_chars limit"""
        if len(text) <= max_chars:
            return text
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines)

    categories_wrapped = [wrap_label(cat) for cat in categories_readable]

    ngo_scores = mean_scores.loc['NGO (Non-governmental organisation)'].values.tolist() if 'NGO (Non-governmental organisation)' in mean_scores.index else []
    trade_union_scores = mean_scores.loc['Trade Union'].values.tolist() if 'Trade Union' in mean_scores.index else []

    # Close the loop for radar chart (repeat first value at end)
    categories_closed = categories_wrapped + [categories_wrapped[0]]
    ngo_scores_closed = ngo_scores + [ngo_scores[0]] if ngo_scores else []
    trade_union_scores_closed = trade_union_scores + [trade_union_scores[0]] if trade_union_scores else []

    # Create radar chart
    fig = go.Figure()

    # Add NGO trace
    fig.add_trace(go.Scatterpolar(
        r=ngo_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='NGO',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    # Add Trade Union trace
    fig.add_trace(go.Scatterpolar(
        r=trade_union_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='Trade Union',
        line=dict(color='#ff7f0e', width=2),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Changed from [1, 5] to [0, 5] to include 'No opinion' = 0
                tickmode='linear',
                tick0=0,
                dtick=1,
                showticklabels=True,
                ticks='outside'
            ),
            angularaxis=dict(
                tickfont=dict(size=9)
            )
        ),
        title={
            'text': 'Comparison of NGO vs Trade Union Priorities<br><sub>Mean Scores Across 6 Actions (Likert Scale 0-5, where 0=No opinion)</sub>',
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            x=0.85,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        width=1200,
        height=1200,
        margin=dict(l=200, r=200, t=120, b=200)  # Further increased margins for wrapped text
    )

    fig.show()

    # Export high-resolution version
    fig.write_image('radar_chart_ngo_vs_trade_union.png', scale=3)
    print("\nRadar chart exported to 'radar_chart_ngo_vs_trade_union.png'")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for org_type in mean_scores.index:
        scores = mean_scores.loc[org_type].values
        print(f"\n{org_type}:")
        print(f"  Mean across all actions: {scores.mean():.2f}")
        print(f"  Std Dev: {scores.std():.2f}")
        print(f"  Min: {scores.min():.2f}")
        print(f"  Max: {scores.max():.2f}")
        print(f"  Range: {scores.max() - scores.min():.2f}")

    # %% Cell 30
    # Extract keywords and bigrams from S2EOT1OE column with aggressive filtering
    from sklearn.feature_extraction.text import CountVectorizer

    import re

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S2EOT1OE']].copy()
    df_text = df_text[df_text['S2EOT1OE'].notna()].copy()
    df_text['S2EOT1OE'] = df_text['S2EOT1OE'].astype(str).str.strip()
    df_text = df_text[df_text['S2EOT1OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S2EOT1OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S2EOT1OE'] = df_text['S2EOT1OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S2EOT1OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S2EOT1OE', 'Resp_Key_S2EOT1OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 31
    LIKERT_MAP = {
        '5 - Very important': 5,
        '4 - Important': 4,
        '3 - Neutral': 3,
        '2 - Not important': 2,
        '1 - Not important at all': 1,
        'No opinion': 0,  # Added to handle "No opinion" responses
    }

    def convert_likert(df, columns):
        for col in columns:
            if col not in df.columns:
                continue
            # Convert to string and strip whitespace
            temp = df[col].astype(str).str.strip()
            # Try exact replacement first
            temp = temp.replace(LIKERT_MAP)
            # Extract leading digit for anything still a string (handles variations in labels)
            mask = temp.apply(lambda x: isinstance(x, str))
            if mask.any():
                temp.loc[mask] = temp.loc[mask].str.extract(r'^(\d)', expand=False)
            # Convert to float
            df[col] = pd.to_numeric(temp, errors='coerce')
        return df

    LIKERT_COLUMNS_all = ['S1EOE11', 'S1EOE12', 'S1EOE13', 'S1EOE14', 'S1EOE15', 'S1EOE16', 
                           'S1EOE21', 'S1EOE22', 'S1EOE23', 'S1EOE24', 'S1EOE25', 'S1EOE26', 
                           'S1EOE31', 'S1EOE32', 'S1EOE33', 'S1EOE41', 'S1EOE42', 'S1EOE43', 
                           'S1EOE44', 'S1EOE45', 'S2EOT11', 'S2EOT12', 'S2EOT13', 'S2EOT14', 
                           'S2EOT15', 'S2EOT16', 'S2EOT51', 'S2EOT52', 'S2EOT53', 'S2EOT54', 
                           'S2EOT55', 'S2EOT56']

    # Reload fresh data and apply conversion
    survey_df_renamed = pd.read_excel('Phase1_SurveyAnalysis_renamed.xlsx', engine='openpyxl')
    survey_df_renamed = convert_likert(survey_df_renamed, LIKERT_COLUMNS_all)

    # Test the conversion
    print("✓ Conversion complete! Sample results:")
    print("\nValue counts for S1EOE11:")
    print(survey_df_renamed['S1EOE11'].value_counts().sort_index())
    print("\nValue counts for S2EOT56:")
    print(survey_df_renamed['S2EOT56'].value_counts().sort_index())

    # Check for any remaining NaN values
    total_values = len(survey_df_renamed) * len(LIKERT_COLUMNS_all)
    nan_count = survey_df_renamed[LIKERT_COLUMNS_all].isna().sum().sum()
    print(f"\n📊 Overall conversion success:")
    print(f"Total cells: {total_values}")
    print(f"NaN values: {nan_count}")
    print(f"Successfully converted: {total_values - nan_count} ({(total_values - nan_count)/total_values*100:.1f}%)")

    # %% Cell 32
    # Diagnostic: Compare all Likert columns to find missing/different values
    print("=== LIKERT COLUMNS VALUE COMPARISON ===\n")

    # Reload fresh data
    survey_df_test = pd.read_excel('Phase1_SurveyAnalysis_renamed.xlsx', engine='openpyxl')

    LIKERT_COLUMNS_all = ['S1EOE11', 'S1EOE12', 'S1EOE13', 'S1EOE14', 'S1EOE15', 'S1EOE16', 
                           'S1EOE21', 'S1EOE22', 'S1EOE23', 'S1EOE24', 'S1EOE25', 'S1EOE26', 
                           'S1EOE31', 'S1EOE32', 'S1EOE33', 'S1EOE41', 'S1EOE42', 'S1EOE43', 
                           'S1EOE44', 'S1EOE45', 'S2EOT11', 'S2EOT12', 'S2EOT13', 'S2EOT14', 
                           'S2EOT15', 'S2EOT16', 'S2EOT51', 'S2EOT52', 'S2EOT53', 'S2EOT54', 
                           'S2EOT55', 'S2EOT56']

    LIKERT_MAP = {
        '5 - Very important': 5,
        '4 - Important': 4,
        '3 - Neutral': 3,
        '2 - Not important': 2,
        '1 - Not important at all': 1,
        'No opinion': 0
    }

    # Collect all unique values across all columns
    all_unique_values = set()
    column_values = {}

    for col in LIKERT_COLUMNS_all:
        if col in survey_df_test.columns:
            unique_vals = survey_df_test[col].dropna().astype(str).str.strip().unique()
            column_values[col] = set(unique_vals)
            all_unique_values.update(unique_vals)

    # Create comparison DataFrame
    comparison_data = []
    for value in sorted(all_unique_values):
        row = {
            'Value': value,
            'In LIKERT_MAP': '✓' if value in LIKERT_MAP else '✗',
            'Column Count': sum(1 for col_vals in column_values.values() if value in col_vals),
            'Appears In': ', '.join([col for col, vals in column_values.items() if value in vals][:3]) + 
                          (f' ... +{sum(1 for col_vals in column_values.values() if value in col_vals) - 3}' 
                           if sum(1 for col_vals in column_values.values() if value in col_vals) > 3 else '')
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    print("\n📊 Summary of all unique values found across Likert columns:")
    print(f"Total unique values: {len(all_unique_values)}")
    print(f"Values in LIKERT_MAP: {sum(1 for v in all_unique_values if v in LIKERT_MAP)}")
    print(f"Values NOT in LIKERT_MAP: {sum(1 for v in all_unique_values if v not in LIKERT_MAP)}")

    print("\n" + "="*80)
    display(comparison_df)

    # Show values NOT in LIKERT_MAP separately for clarity
    missing_values = comparison_df[comparison_df['In LIKERT_MAP'] == '✗']
    if not missing_values.empty:
        print("\n⚠️ VALUES NOT IN LIKERT_MAP (need to be added):")
        print("="*80)
        display(missing_values)

    # %% Cell 33
    # Diagnostic: Check what the NaN values represent
    print("=== ANALYZING NaN VALUES ===\n")

    # Load original data to compare
    survey_df_original = pd.read_excel('Phase1_SurveyAnalysis_renamed.xlsx', engine='openpyxl')

    # Check NaN counts before vs after conversion
    print("NaN counts comparison:")
    print(f"{'Column':<12} {'Original NaN':<15} {'After Conversion':<20} {'Difference'}")
    print("="*65)

    for col in LIKERT_COLUMNS_all:
        if col in survey_df_original.columns and col in survey_df_renamed.columns:
            original_nan = survey_df_original[col].isna().sum()
            converted_nan = survey_df_renamed[col].isna().sum()
            diff = converted_nan - original_nan
            print(f"{col:<12} {original_nan:<15} {converted_nan:<20} {diff}")

    # Summary
    original_total_nan = survey_df_original[LIKERT_COLUMNS_all].isna().sum().sum()
    converted_total_nan = survey_df_renamed[LIKERT_COLUMNS_all].isna().sum().sum()

    print("\n" + "="*65)
    print(f"Total NaN in original data: {original_total_nan}")
    print(f"Total NaN after conversion: {converted_total_nan}")
    print(f"New NaN from failed conversions: {converted_total_nan - original_total_nan}")

    print("\n📝 Interpretation:")
    if converted_total_nan - original_total_nan == 0:
        print("✓ All NaN values are from originally empty cells (respondents who skipped questions)")
        print("✓ No conversion failures - all populated cells were successfully converted!")
    else:
        print(f"⚠ {converted_total_nan - original_total_nan} cells failed to convert")
        
    print("\n💡 For mean calculations:")
    print("Pandas .mean() automatically ignores NaN values (skipna=True by default),")
    print("so means will be calculated only from actual responses. This is correct!")

    # %% Cell 34
    # Calculate mean scores for NGOs and Trade Unions separately
    import plotly.graph_objects as go



    # Define the Likert columns and organization type
    likert_columns = ['S2EOT51', 'S2EOT52', 'S2EOT53', 'S2EOT54', 'S2EOT55', 'S2EOT56']
    org_type_col = 'I2'

    # Calculate mean scores by organization type
    mean_scores = survey_df_renamed.groupby(org_type_col)[likert_columns].mean()

    print("="*60)
    print("MEAN SCORES BY ORGANIZATION TYPE")
    print("="*60)
    display(mean_scores)

    # Calculate priority gap: mean(NGO) - mean(TU)
    if 'NGO (Non-governmental organisation)' in mean_scores.index and 'Trade Union' in mean_scores.index:
        priority_gap = mean_scores.loc['NGO (Non-governmental organisation)'] - mean_scores.loc['Trade Union']
        
        print("\n" + "="*60)
        print("PRIORITY GAP ANALYSIS")
        print("NGO mean - Trade Union mean")
        print("="*60)
        print("Positive values = NGOs prioritize more | Negative values = Trade Unions prioritize more")
        print()
        
        # Create DataFrame with readable labels for better display
        gap_df = pd.DataFrame({
            'Metric': [label_mapping.get(col, col) for col in likert_columns],
            'NGO Mean': mean_scores.loc['NGO (Non-governmental organisation)'].values,
            'TU Mean': mean_scores.loc['Trade Union'].values,
            'Priority Gap (NGO - TU)': priority_gap.values
        })
        gap_df = gap_df.sort_values('Priority Gap (NGO - TU)', ascending=False)
        display(gap_df)
        
        # Summary
        print(f"\nLargest disagreements:")
        print(f"  NGOs prioritize more: {gap_df.iloc[0]['Metric']} (gap: +{gap_df.iloc[0]['Priority Gap (NGO - TU)']:.2f})")
        print(f"  Trade Unions prioritize more: {gap_df.iloc[-1]['Metric']} (gap: {gap_df.iloc[-1]['Priority Gap (NGO - TU)']:.2f})")
        print(f"\nMean absolute gap: {gap_df['Priority Gap (NGO - TU)'].abs().mean():.2f}")

    # Prepare data for radar chart with readable labels
    categories_codes = likert_columns
    categories_readable = [label_mapping.get(col, col) for col in categories_codes]  # Use readable names

    # Wrap long labels to multiple lines for better display
    def wrap_label(text, max_chars=40):
        """Wrap text at spaces near max_chars limit"""
        if len(text) <= max_chars:
            return text
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines)

    categories_wrapped = [wrap_label(cat) for cat in categories_readable]

    ngo_scores = mean_scores.loc['NGO (Non-governmental organisation)'].values.tolist() if 'NGO (Non-governmental organisation)' in mean_scores.index else []
    trade_union_scores = mean_scores.loc['Trade Union'].values.tolist() if 'Trade Union' in mean_scores.index else []

    # Close the loop for radar chart (repeat first value at end)
    categories_closed = categories_wrapped + [categories_wrapped[0]]
    ngo_scores_closed = ngo_scores + [ngo_scores[0]] if ngo_scores else []
    trade_union_scores_closed = trade_union_scores + [trade_union_scores[0]] if trade_union_scores else []

    # Create radar chart
    fig = go.Figure()

    # Add NGO trace
    fig.add_trace(go.Scatterpolar(
        r=ngo_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='NGO',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    # Add Trade Union trace
    fig.add_trace(go.Scatterpolar(
        r=trade_union_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='Trade Union',
        line=dict(color='#ff7f0e', width=2),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Changed from [1, 5] to [0, 5] to include 'No opinion' = 0
                tickmode='linear',
                tick0=0,
                dtick=1,
                showticklabels=True,
                ticks='outside'
            ),
            angularaxis=dict(
                tickfont=dict(size=9)
            )
        ),
        title={
            'text': 'Comparison of NGO vs Trade Union Priorities<br><sub>Mean Scores Across 6 Actions (Likert Scale 0-5, where 0=No opinion)</sub>',
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            x=0.85,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        width=1200,
        height=1200,
        margin=dict(l=200, r=200, t=120, b=200)  # Further increased margins for wrapped text
    )

    fig.show()

    # Export high-resolution version
    fig.write_image('radar_chart_ngo_vs_trade_union_image2.png', scale=3)
    print("\nRadar chart exported to 'radar_chart_ngo_vs_trade_union_image2.png'")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for org_type in mean_scores.index:
        scores = mean_scores.loc[org_type].values
        print(f"\n{org_type}:")
        print(f"  Mean across all actions: {scores.mean():.2f}")
        print(f"  Std Dev: {scores.std():.2f}")
        print(f"  Min: {scores.min():.2f}")
        print(f"  Max: {scores.max():.2f}")
        print(f"  Range: {scores.max() - scores.min():.2f}")

    # %% Cell 35
    ### Analysing Importance of Six Actions in white paper of AI

    # %% Cell 36
    # Calculate mean scores for NGOs and Trade Unions separately
    import plotly.graph_objects as go

    # Create mapping from Data Reference ID to readable Column Names
    label_mapping = dict(zip(
        comments_df['Data Reference ID'].astype(str).str.strip(),
        comments_df['Column Names'].astype(str).str.strip()
    ))

    # Define the Likert columns and organization type
    likert_columns = ['S1EOE11', 'S1EOE12', 'S1EOE13', 'S1EOE14', 'S1EOE15', 'S1EOE16']
    org_type_col = 'I2'

    # Calculate mean scores by organization type
    mean_scores = survey_df_renamed.groupby(org_type_col)[likert_columns].mean()

    print("="*60)
    print("MEAN SCORES BY ORGANIZATION TYPE")
    print("="*60)
    display(mean_scores)

    # Calculate priority gap: mean(NGO) - mean(TU)
    if 'NGO (Non-governmental organisation)' in mean_scores.index and 'Trade Union' in mean_scores.index:
        priority_gap = mean_scores.loc['NGO (Non-governmental organisation)'] - mean_scores.loc['Trade Union']
        
        print("\n" + "="*60)
        print("PRIORITY GAP ANALYSIS")
        print("NGO mean - Trade Union mean")
        print("="*60)
        print("Positive values = NGOs prioritize more | Negative values = Trade Unions prioritize more")
        print()
        
        # Create DataFrame with readable labels for better display
        gap_df = pd.DataFrame({
            'Metric': [label_mapping.get(col, col) for col in likert_columns],
            'NGO Mean': mean_scores.loc['NGO (Non-governmental organisation)'].values,
            'TU Mean': mean_scores.loc['Trade Union'].values,
            'Priority Gap (NGO - TU)': priority_gap.values
        })
        gap_df = gap_df.sort_values('Priority Gap (NGO - TU)', ascending=False)
        display(gap_df)
        
        # Summary
        print(f"\nLargest disagreements:")
        print(f"  NGOs prioritize more: {gap_df.iloc[0]['Metric']} (gap: +{gap_df.iloc[0]['Priority Gap (NGO - TU)']:.2f})")
        print(f"  Trade Unions prioritize more: {gap_df.iloc[-1]['Metric']} (gap: {gap_df.iloc[-1]['Priority Gap (NGO - TU)']:.2f})")
        print(f"\nMean absolute gap: {gap_df['Priority Gap (NGO - TU)'].abs().mean():.2f}")

    # Prepare data for radar chart with readable labels
    categories_codes = likert_columns
    categories_readable = [label_mapping.get(col, col) for col in categories_codes]  # Use readable names

    # Wrap long labels to multiple lines for better display
    def wrap_label(text, max_chars=40):
        """Wrap text at spaces near max_chars limit"""
        if len(text) <= max_chars:
            return text
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '<br>'.join(lines)

    categories_wrapped = [wrap_label(cat) for cat in categories_readable]

    ngo_scores = mean_scores.loc['NGO (Non-governmental organisation)'].values.tolist() if 'NGO (Non-governmental organisation)' in mean_scores.index else []
    trade_union_scores = mean_scores.loc['Trade Union'].values.tolist() if 'Trade Union' in mean_scores.index else []

    # Close the loop for radar chart (repeat first value at end)
    categories_closed = categories_wrapped + [categories_wrapped[0]]
    ngo_scores_closed = ngo_scores + [ngo_scores[0]] if ngo_scores else []
    trade_union_scores_closed = trade_union_scores + [trade_union_scores[0]] if trade_union_scores else []

    # Create radar chart
    fig = go.Figure()

    # Add NGO trace
    fig.add_trace(go.Scatterpolar(
        r=ngo_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='NGO',
        line=dict(color='#1f77b4', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    # Add Trade Union trace
    fig.add_trace(go.Scatterpolar(
        r=trade_union_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='Trade Union',
        line=dict(color='#ff7f0e', width=2),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Changed from [1, 5] to [0, 5] to include 'No opinion' = 0
                tickmode='linear',
                tick0=0,
                dtick=1,
                showticklabels=True,
                ticks='outside'
            ),
            angularaxis=dict(
                tickfont=dict(size=9)
            )
        ),
        title={
            'text': 'Comparison of NGO vs Trade Union Priorities - Importance of AI Actions in 6 sections of White Paper<br><sub>Mean Scores Across 6 Actions (Likert Scale 0-5, where 0=No opinion)</sub>',
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            x=0.85,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        width=1200,
        height=1200,
        margin=dict(l=200, r=200, t=120, b=200)  # Further increased margins for wrapped text
    )

    fig.show()

    # Export high-resolution version
    fig.write_image('radar_chart_ngo_vs_trade_union_image3_Importance of AI actions.png', scale=3)
    print("\nRadar chart exported to 'radar_chart_ngo_vs_trade_union_image3_Importance of AI actions.png'")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for org_type in mean_scores.index:
        scores = mean_scores.loc[org_type].values
        print(f"\n{org_type}:")
        print(f"  Mean across all actions: {scores.mean():.2f}")
        print(f"  Std Dev: {scores.std():.2f}")
        print(f"  Min: {scores.min():.2f}")
        print(f"  Max: {scores.max():.2f}")
        print(f"  Range: {scores.max() - scores.min():.2f}")

    # %% Cell 37
    # Extract keywords and bigrams from S1EOE1OE column with aggressive filtering
    from sklearn.feature_extraction.text import CountVectorizer

    import re

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S1EOE1OE']].copy()
    df_text = df_text[df_text['S1EOE1OE'].notna()].copy()
    df_text['S1EOE1OE'] = df_text['S1EOE1OE'].astype(str).str.strip()
    df_text = df_text[df_text['S1EOE1OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S1EOE1OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S1EOE1OE'] = df_text['S1EOE1OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S1EOE1OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S1EOE1OE', 'Resp_Key_S1EOE1OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 38
    ### Analysing categorical data to understand Sentiment Shifts towards the new legislation

    # %% Cell 39
    # Create categorical data type for S2EOT2 with ordered categories
    from pandas.api.types import CategoricalDtype

    # Define the categorical order
    category_order = [
        'Current legislation is fully sufficient',
        'Current legislation may have some gaps',
        'There is a need for a new legislation',
        'Other',
        'No opinion'
    ]

    # Create categorical dtype with ordering
    cat_type = CategoricalDtype(categories=category_order, ordered=True)

    # Apply the categorical type to the column
    survey_df_renamed['S2EOT2'] = survey_df_renamed['S2EOT2'].astype(cat_type)

    # Verify the conversion
    print("Column 'S2EOT2' converted to ordered categorical")
    print(f"\nData type: {survey_df_renamed['S2EOT2'].dtype}")
    print(f"\nCategories: {survey_df_renamed['S2EOT2'].cat.categories.tolist()}")
    print(f"Ordered: {survey_df_renamed['S2EOT2'].cat.ordered}")

    # Show value counts in the defined order
    print("\nValue counts (in categorical order):")
    print(survey_df_renamed['S2EOT2'].value_counts().sort_index())

    # %% Cell 40
    # Create contingency table: Organization Type vs Legislation Sentiment
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT2'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs Legislation Sentiment (S2EOT2)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT2'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 41
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S2EOT2 from comments_df
    s2eot2_label = label_mapping.get('S2EOT2', 'S2EOT2')
    print(f"S2EOT2 corresponds to: '{s2eot2_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999']  # Gray tones for 'Other' and 'No opinion'
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Views on Legislation Sufficiency by Organization Type\n{s2eot2_label}', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='Legislation Sentiment',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_legislation_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_legislation_sentiment.png'")

    # %% Cell 42
    # Extract keywords and bigrams from S2EOT2OE column with aggressive filtering
    from sklearn.feature_extraction.text import CountVectorizer

    import re

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S2EOT2OE']].copy()
    df_text = df_text[df_text['S2EOT2OE'].notna()].copy()
    df_text['S2EOT2OE'] = df_text['S2EOT2OE'].astype(str).str.strip()
    df_text = df_text[df_text['S2EOT2OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S2EOT2OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S2EOT2OE'] = df_text['S2EOT2OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S2EOT2OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S2EOT2OE', 'Resp_Key_S2EOT2OE']].copy()

    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 43
    # Create contingency table: Organization Type vs Legislation Sentiment
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT3'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs Need for new AI Rules (S2EOT3)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT3'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 44
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S2EOT3 from comments_df
    s2eot3_label = label_mapping.get('S2EOT3', 'S2EOT3')
    print(f"S2EOT3 corresponds to: '{s2eot3_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999']  # Gray tones for 'Other' and 'No opinion'
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Views on WHhether new rules are needed for future AI systems by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='Legislation Sentiment',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_NewRulesAISystems_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_NewRulesAISystems_sentiment.png'")

    # %% Cell 45
    # Extract keywords and bigrams from S2EOT3OE column with aggressive filtering

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S2EOT3OE']].copy()
    df_text = df_text[df_text['S2EOT3OE'].notna()].copy()
    df_text['S2EOT3OE'] = df_text['S2EOT3OE'].astype(str).str.strip()
    df_text = df_text[df_text['S2EOT3OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S2EOT3OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S2EOT3OE'] = df_text['S2EOT3OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S2EOT3OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S2EOT3OE', 'Resp_Key_S2EOT3OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 46
    # Create contingency table: Organization Type vs Appetite to determine High Risk AI applications
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT4'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs Approach Determining High Risk AI Application (S2EOT4)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT4'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 47
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S2EOT4 from comments_df
    s2eot4_label = label_mapping.get('S2EOT4', 'S2EOT4')
    print(f"S2EOT4 corresponds to: '{s2eot4_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999']  # Gray tones for 'Other' and 'No opinion'
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Views on Approach Determining High Risk AI Applications by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='Legislation Sentiment',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_ApproachToHighRiskAIApps_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_ApproachToHighRiskAIApps_sentiment.png'")

    # %% Cell 48
    # Extract keywords and bigrams from S2EOT4OE1 and S2EOT4OE2 columns with aggressive filtering

    # Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Function to process a single column
    def process_column(df, column_name, custom_stop_words):
        """Process a single text column for keyword extraction"""
        
        # Step 1: Prepare the data - filter out null/empty responses
        df_text = df[['REF', 'I2', column_name]].copy()
        df_text = df_text[df_text[column_name].notna()].copy()
        df_text[column_name] = df_text[column_name].astype(str).str.strip()
        df_text = df_text[df_text[column_name] != '']
        
        # Step 2: Use CountVectorizer with aggressive thresholds
        vectorizer = CountVectorizer(
            strip_accents='unicode',      # Normalize European characters
            stop_words=custom_stop_words, # Filter specific noise words
            max_df=0.7,                   # Exclude words appearing in >70% of docs
            min_df=2,                     # Must appear in at least 2 docs
            ngram_range=(1, 2),           # Surface unigrams and bigrams
            lowercase=True
        )
        
        # Fit and extract frequencies
        X = vectorizer.fit_transform(df_text[column_name])
        feature_names = vectorizer.get_feature_names_out()
        term_freq = X.sum(axis=0).A1 
        term_freq_dict = dict(zip(feature_names, term_freq))
        
        # Step 3: Identify TOP 50 most frequent terms to allow Bigrams to surface
        top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
        top_50_list = [keyword for keyword, freq in top_50_keywords]
        
        # Step 4: Keyword Matching
        resp_key_col = f'Resp_Key_{column_name}'
        df_text[resp_key_col] = df_text[column_name].apply(
            lambda x: extract_response_keywords(x, top_50_list)
        )
        
        # Step 5: Organization - Sort by User Type (I2) and then by Response Length
        df_text['response_length'] = df_text[column_name].str.len()
        df_final = df_text.sort_values(
            by=['I2', 'response_length'], 
            ascending=[True, False]
        )[['REF', 'I2', column_name, resp_key_col]].copy()
        
        return df_final, top_50_keywords

    # Process both columns
    print("="*80)
    print("PROCESSING S2EOT4OE1")
    print("="*80)
    df_final_oe1, top_50_oe1 = process_column(survey_df_renamed, 'S2EOT4OE1', custom_stop_words)
    print(f"✓ Success! {len(df_final_oe1)} responses processed for S2EOT4OE1.")
    print("\nTop 10 Detected Phrases for S2EOT4OE1:")
    for k, v in top_50_oe1[:10]:
        print(f"  - {k} ({int(v)})")

    print("\n" + "="*80)
    print("PROCESSING S2EOT4OE2")
    print("="*80)
    df_final_oe2, top_50_oe2 = process_column(survey_df_renamed, 'S2EOT4OE2', custom_stop_words)
    print(f"✓ Success! {len(df_final_oe2)} responses processed for S2EOT4OE2.")
    print("\nTop 10 Detected Phrases for S2EOT4OE2:")
    for k, v in top_50_oe2[:10]:
        print(f"  - {k} ({int(v)})")

    # Display side-by-side summary
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    print(f"\nS2EOT4OE1: {len(df_final_oe1)} responses | S2EOT4OE2: {len(df_final_oe2)} responses")

    # Option 1: Copy S2EOT4OE1 to clipboard
    print("\n" + "="*80)
    print("COPYING S2EOT4OE1 TO CLIPBOARD")
    print("="*80)
    df_final_oe1.to_clipboard(excel=True, index=False)
    print("✓ S2EOT4OE1 data is now on your clipboard. Paste it into your Excel sheet.")

    # Display first few rows of each for preview
    print("\n" + "="*80)
    print("PREVIEW: S2EOT4OE1 (First 3 rows)")
    print("="*80)
    display(df_final_oe1.head(3))

    print("\n" + "="*80)
    print("PREVIEW: S2EOT4OE2 (First 3 rows)")
    print("="*80)
    display(df_final_oe2.head(3))

    # Store results for later use
    print("\n" + "="*80)
    print("NOTE: Results stored in df_final_oe1 and df_final_oe2")
    print("To copy S2EOT4OE2 to clipboard, run: df_final_oe2.to_clipboard(excel=True, index=False)")
    print("="*80)

    # %% Cell 49
    df_final_oe2.to_clipboard(excel=True, index=False)

    # %% Cell 50
    # Create contingency table: Organization Type vs Appetite for Biometric ID systems in public spaces
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT6'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs Appetite for Biometric ID Systems in Public Spaces (S2EOT6)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT6'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 51
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S2EOT4 from comments_df
    s2eot6_label = label_mapping.get('S2EOT6', 'S2EOT6')
    print(f"S2EOT6 corresponds to: '{s2eot6_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    # Define colors: Yellow-Green-Blue gradient for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999', '#E5E5E5']  # Gray tones for 'Other', 'No opinion', etc.
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Views on Appetite for Biometrics Systems in Public by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='Biometrics Sentiment',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_Biometrics_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_Biometrics_sentiment.png'")

    # %% Cell 52
    # Extract keywords and bigrams from S2EOT6OE column with aggressive filtering

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S2EOT6OE']].copy()
    df_text = df_text[df_text['S2EOT6OE'].notna()].copy()
    df_text['S2EOT6OE'] = df_text['S2EOT6OE'].astype(str).str.strip()
    df_text = df_text[df_text['S2EOT6OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S2EOT6OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S2EOT6OE'] = df_text['S2EOT6OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S2EOT6OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S2EOT6OE', 'Resp_Key_S2EOT6OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 53
    # Create contingency table: Organization Type vs European Values and AI
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT8'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs  European Values & AI  (S2EOT8)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S2EOT8'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 54
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S2EOT4 from comments_df
    s2eot8_label = label_mapping.get('S2EOT8', 'S2EOT8')
    print(f"S2EOT8 corresponds to: '{s2eot8_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    # Define colors: Yellow-Green-Blue gradient for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999', '#E5E5E5']  # Gray tones for 'Other', 'No opinion', etc.
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Views on Europian Values & AI by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='EU Values & AI Sentiment',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_EUValues_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_EUValues_sentiment.png'")

    # %% Cell 55
    # Extract keywords and bigrams from S2EOT8OE1 and S2EOT8OE2 columns with aggressive filtering

    # Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Function to process a single column
    def process_column(df, column_name, custom_stop_words):
        """Process a single text column for keyword extraction"""
        
        # Step 1: Prepare the data - filter out null/empty responses
        df_text = df[['REF', 'I2', column_name]].copy()
        df_text = df_text[df_text[column_name].notna()].copy()
        df_text[column_name] = df_text[column_name].astype(str).str.strip()
        df_text = df_text[df_text[column_name] != '']
        
        # Step 2: Use CountVectorizer with aggressive thresholds
        vectorizer = CountVectorizer(
            strip_accents='unicode',      # Normalize European characters
            stop_words=custom_stop_words, # Filter specific noise words
            max_df=0.7,                   # Exclude words appearing in >70% of docs
            min_df=2,                     # Must appear in at least 2 docs
            ngram_range=(1, 2),           # Surface unigrams and bigrams
            lowercase=True
        )
        
        # Fit and extract frequencies
        X = vectorizer.fit_transform(df_text[column_name])
        feature_names = vectorizer.get_feature_names_out()
        term_freq = X.sum(axis=0).A1 
        term_freq_dict = dict(zip(feature_names, term_freq))
        
        # Step 3: Identify TOP 50 most frequent terms to allow Bigrams to surface
        top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
        top_50_list = [keyword for keyword, freq in top_50_keywords]
        
        # Step 4: Keyword Matching
        resp_key_col = f'Resp_Key_{column_name}'
        df_text[resp_key_col] = df_text[column_name].apply(
            lambda x: extract_response_keywords(x, top_50_list)
        )
        
        # Step 5: Organization - Sort by User Type (I2) and then by Response Length
        df_text['response_length'] = df_text[column_name].str.len()
        df_final = df_text.sort_values(
            by=['I2', 'response_length'], 
            ascending=[True, False]
        )[['REF', 'I2', column_name, resp_key_col]].copy()
        
        return df_final, top_50_keywords

    # Process both columns
    print("="*80)
    print("PROCESSING S2EOT8OE1")
    print("="*80)
    df_final_oe1, top_50_oe1 = process_column(survey_df_renamed, 'S2EOT8OE1', custom_stop_words)
    print(f"✓ Success! {len(df_final_oe1)} responses processed for S2EOT8OE1.")
    print("\nTop 10 Detected Phrases for S2EOT8OE1:")
    for k, v in top_50_oe1[:10]:
        print(f"  - {k} ({int(v)})")

    print("\n" + "="*80)
    print("PROCESSING S2EOT8OE2")
    print("="*80)
    df_final_oe2, top_50_oe2 = process_column(survey_df_renamed, 'S2EOT8OE2', custom_stop_words)
    print(f"✓ Success! {len(df_final_oe2)} responses processed for S2EOT8OE2.")
    print("\nTop 10 Detected Phrases for S2EOT8OE2:")
    for k, v in top_50_oe2[:10]:
        print(f"  - {k} ({int(v)})")

    # Display side-by-side summary
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    print(f"\nS2EOT8OE1: {len(df_final_oe1)} responses | S2EOT8OE2: {len(df_final_oe2)} responses")

    # Option 1: Copy S2EOT8OE1 to clipboard
    print("\n" + "="*80)
    print("COPYING S2EOT8OE1 TO CLIPBOARD")
    print("="*80)
    df_final_oe1.to_clipboard(excel=True, index=False)
    print("✓ S2EOT8OE1 data is now on your clipboard. Paste it into your Excel sheet.")

    # Display first few rows of each for preview
    print("\n" + "="*80)
    print("PREVIEW: S2EOT8OE1 (First 3 rows)")
    print("="*80)
    display(df_final_oe1.head(3))

    print("\n" + "="*80)
    print("PREVIEW: S2EOT8OE2 (First 3 rows)")
    print("="*80)
    display(df_final_oe2.head(3))

    # Store results for later use
    print("\n" + "="*80)
    print("NOTE: Results stored in df_final_oe1 and df_final_oe2")
    print("To copy S2EOT8OE2 to clipboard, run: df_final_oe2.to_clipboard(excel=True, index=False)")
    print("="*80)

    # %% Cell 56
    df_final_oe2.to_clipboard(excel=True, index=False)

    # %% Cell 57
    # Create contingency table: Organization Type vs Legal needs for Risks from AI Use 
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S3SLI1'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs Legal Needs for Risks from AI (S3SLI1)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S3SLI1'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 58
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S3SLI1 from comments_df
    s3LI1_label = label_mapping.get('S3SLI1', 'S3SLI1')
    print(f"S3SLI1 corresponds to: '{s3LI1_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999']  # Gray tones for 'Other' and 'No opinion'
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Views on Legal Needs from AI Risks by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='Legal Risks from AI Sentiment',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_LegalAI_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_LegalAI_sentiment.png'")

    # %% Cell 59
        # Extract keywords and bigrams from S3SLI1OE column with aggressive filtering

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S3SLI1OE']].copy()
    df_text = df_text[df_text['S3SLI1OE'].notna()].copy()
    df_text['S3SLI1OE'] = df_text['S3SLI1OE'].astype(str).str.strip()
    df_text = df_text[df_text['S3SLI1OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S3SLI1OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S3SLI1OE'] = df_text['S3SLI1OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S3SLI1OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S3SLI1OE', 'Resp_Key_S3SLI1OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 60
    # Create contingency table: Organization Type vs Legal needs for Risks from AI Use 
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S3SLI2'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs New Risk Assessment Framework (S3SLI2)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S3SLI2'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 61
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S3SLI1 from comments_df
    s3LI2_label = label_mapping.get('S3SLI2', 'S3SLI2')
    print(f"S3SLI2 corresponds to: '{s3LI2_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999']  # Gray tones for 'Other' and 'No opinion'
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'New Risk Assessment Framework by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='New Risk Assessment Framework',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_S3SLI2_NewRiskAssess_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_S3SLI2_NewRiskAssess_sentiment.png'")

    # %% Cell 62
        # Extract keywords and bigrams from S3SLI2OE column with aggressive filtering

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S3SLI2OE']].copy()
    df_text = df_text[df_text['S3SLI2OE'].notna()].copy()
    df_text['S3SLI2OE'] = df_text['S3SLI2OE'].astype(str).str.strip()
    df_text = df_text[df_text['S3SLI2OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use CountVectorizer with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S3SLI2OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S3SLI2OE'] = df_text['S3SLI2OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S3SLI2OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S3SLI2OE', 'Resp_Key_S3SLI2OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")

    # %% Cell 63
    # Create contingency table: Organization Type vs Need for Updated EU Legislative Framework to assess risk from AI 
    # Normalized by row (index) so each organization type sums to 100%

    # Create contingency table with raw counts
    contingency_table_raw = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S3SLI3'],
        margins=True,
        margins_name='Total'
    )

    print("="*80)
    print("CONTINGENCY TABLE - RAW COUNTS")
    print("Organization Type (I2) vs Updating EU Legistlative Framework to assess AI Risk (S3SLI3)")
    print("="*80)
    display(contingency_table_raw)

    # Create normalized contingency table (row percentages)
    contingency_table_pct = pd.crosstab(
        survey_df_renamed['I2'], 
        survey_df_renamed['S3SLI3'],
        normalize='index'  # Each row sums to 1.0 (100%)
    ) * 100  # Convert to percentages

    print("\n" + "="*80)
    print("CONTINGENCY TABLE - ROW PERCENTAGES")
    print("Each row shows percentage distribution for that organization type")
    print("="*80)
    display(contingency_table_pct.round(2))

    # Summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    for org_type in contingency_table_pct.index:
        print(f"\n{org_type}:")
        for response, pct in contingency_table_pct.loc[org_type].items():
            if pct > 0:
                print(f"  {response}: {pct:.1f}%")

    # %% Cell 64
    # Create horizontal stacked bar chart from normalized contingency table
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Get readable column name for S3SLI1 from comments_df
    s3LI3_label = label_mapping.get('S3SLI3', 'S3SLI3')
    print(f"S3SLI3 corresponds to: '{s3LI3_label}'")

    # Define colors: sequential palette for first 3 categories, gray for others
    colors_sequential = plt.cm.YlGnBu(np.linspace(0.3, 0.8, 3))  # Yellow-Green-Blue gradient
    colors_neutral = ['#CCCCCC', '#999999']  # Gray tones for 'Other' and 'No opinion'
    colors = list(colors_sequential) + colors_neutral

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the categories in order
    categories = contingency_table_pct.columns.tolist()
    org_types = contingency_table_pct.index.tolist()

    # Create horizontal stacked bar chart
    left = np.zeros(len(org_types))
    for i, category in enumerate(categories):
        values = contingency_table_pct[category].values
        bars = ax.barh(org_types, values, left=left, 
                       color=colors[i], label=category, edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars (only if > 5% to avoid clutter)
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:  # Only show labels for segments > 5%
                x_pos = left[j] + value / 2
                ax.text(x_pos, j, f'{value:.1f}%', 
                       ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        
        left += values

    # Customize the plot
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organization Type', fontsize=13, fontweight='bold')
    ax.set_title(f'Updating EU Legistlative Framework to assess AI Risk by Organization Type', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Position legend outside the plot on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=10, frameon=True, title='Need for Updating EU Legislative Framework',
             title_fontsize=11, framealpha=0.9)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

    # Export high-resolution version
    fig.savefig('stacked_bar_S3SLI3_UpdatingEULegislation_sentiment.png', dpi=300, bbox_inches='tight')
    print("\nChart exported to 'stacked_bar_S3SLI3_UpdatingEULegislation_sentiment.png'")

    # %% Cell 65
    # Extract keywords and bigrams from S3SLI3OE column with aggressive filtering
    from sklearn.feature_extraction.text import CountVectorizer
    import re

    # Step 1: Prepare the data - filter out null/empty responses
    # Using survey_df_renamed as per your previous setup
    df_text = survey_df_renamed[['REF', 'I2', 'S3SLI3OE']].copy()
    df_text = df_text[df_text['S3SLI3OE'].notna()].copy()
    df_text['S3SLI3OE'] = df_text['S3SLI3OE'].astype(str).str.strip()
    df_text = df_text[df_text['S3SLI3OE'] != '']

    # Step 2: Define Hard Exclusions (Stop Words)
    custom_stop_words = [
        'the', 'on', 'to', 'in', 'and', 'of', 'is', 'as', 'not', 'should', 
        'for', 'with', 'that', 'it', 'by', 'an', 'are', 'this', 'be', 'from', 'a',
        'must', 'have', 'at', 'or', 'if', 'but', 'we', 'they', 'their', 'our', 'us',
        'so', 'will', 'can', 'all', 'any', 'more', 'no', 'one', 'about', 'what', 'when',
        'which', 'there', 'also', 'than', 'other', 'some', 'such', 'may', 'like', 'just', 'only',
        'new', 'use', 'used', 'using', 'due', 'because', 'how', 'these', 'those', 'most', 'many', 'well',
        'even', 'over', 'between', 'during', 'while', 'where', 'who', 'would', 'should', 'could',
        'did', 'does', 'doing', 'after', 'before', 'through', 'each', 'every', 'few', 'further',
        'however', 'therefore', 'thus', 'hence', 'indeed', 'although', 'though', 'yet', 'still',
        'rather', 'quite', 'rather', 'very', 'too', 'much', 'many', 'based'
    ]

    # Step 3: Use 
    #  with aggressive thresholds
    vectorizer = CountVectorizer(
        strip_accents='unicode',    # Normalize European characters
        stop_words=custom_stop_words, # Filter specific noise words
        max_df=0.7,                 # Exclude words appearing in >70% of docs
        min_df=2,                   # Must appear in at least 2 docs
        ngram_range=(1, 2),         # Surface unigrams and bigrams
        lowercase=True
    )

    # Fit and extract frequencies
    X = vectorizer.fit_transform(df_text['S3SLI3OE'])
    feature_names = vectorizer.get_feature_names_out()
    term_freq = X.sum(axis=0).A1 
    term_freq_dict = dict(zip(feature_names, term_freq))

    # Step 4: Identify TOP 50 most frequent terms to allow Bigrams to surface
    top_50_keywords = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_list = [keyword for keyword, freq in top_50_keywords]

    # Step 5: Keyword Matching Function
    def extract_response_keywords(text, top_keywords_set):
        if pd.isna(text) or text.strip() == '':
            return ''
        
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in top_keywords_set:
            # Use word boundaries for unigrams to avoid partial matching (e.g., 'ai' in 'said')
            if len(keyword.split()) == 1:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        
        return ', '.join(matched_keywords)

    # Apply matching
    df_text['Resp_Key_S3SLI3OE'] = df_text['S3SLI3OE'].apply(
        lambda x: extract_response_keywords(x, top_50_list)
    )

    # Step 6: Organization - Sort by User Type (I2) and then by Response Length
    df_text['response_length'] = df_text['S3SLI3OE'].str.len()
    df_final = df_text.sort_values(
        by=['I2', 'response_length'], 
        ascending=[True, False]
    )[['REF', 'I2', 'S3SLI3OE', 'Resp_Key_S3SLI3OE']].copy()
    # Step 7: Output to Clipboard
    df_final.to_clipboard(excel=True, index=False)

    print(f"✓ Success! {len(df_final)} responses processed.")
    print("The data is now on your clipboard. Paste it into your Excel sheet.")
    print("\nTop 10 Detected Phrases for context:")
    for k, v in top_50_keywords[:10]:
        print(f"- {k} ({int(v)})")



if __name__ == "__main__":
    main()


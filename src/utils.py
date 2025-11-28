import pandas as pd
import datetime as dt
import os

def load_and_merge_raw_data(raw_data_path):
    """
    Carga los datasets raw de Olist y realiza el merge inicial.
    
    Args:
        raw_data_path (str): Ruta al directorio donde están los CSVs raw.

    Returns:
        pd.DataFrame: DataFrame fusionado (Orders + Items + Customers).
        None: Si falla la carga.
    """
    try:
        # Definir rutas
        orders_path = os.path.join(raw_data_path, 'olist_orders_dataset.csv')
        items_path = os.path.join(raw_data_path, 'olist_order_items_dataset.csv')
        cust_path = os.path.join(raw_data_path, 'olist_customers_dataset.csv')

        # Cargar
        print("   -> Cargando CSVs...")
        df_orders = pd.read_csv(orders_path)
        df_items = pd.read_csv(items_path)
        df_cust = pd.read_csv(cust_path)

        # Merges
        # 1. Orders + Items (Left join para no perder órdenes sin items si las hubiera)
        temp = df_orders.merge(df_items, on='order_id', how='left')
        # 2. + Customers (Inner join para asegurar ubicación)
        df_master = temp.merge(df_cust, on='customer_id', how='inner')

        return df_master

    except FileNotFoundError as e:
        print(f"❌ Error crítico cargando archivos: {e}")
        return None

def clean_data(df):
    """
    Realiza la limpieza técnica y de negocio del dataset maestro.
    
    Procesos:
    1. Conversión de fechas.
    2. Filtro de estado de orden ('delivered').
    3. Manejo de nulos en fechas clave.

    Args:
        df (pd.DataFrame): DataFrame maestro crudo.

    Returns:
        pd.DataFrame: DataFrame limpio listo para RFM.
    """
    df_clean = df.copy()

    # 1. Conversión de fechas
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                 'order_delivered_customer_date']
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # 2. Filtro de Negocio: Solo entregados
    # Ignoramos cancelados o en proceso para el cálculo de valor real
    df_clean = df_clean[df_clean['order_status'] == 'delivered']

    # 3. Limpieza de Nulos Críticos
    # Si no hay fecha de entrega o ID de cliente único, no sirve
    df_clean = df_clean.dropna(subset=['order_purchase_timestamp', 'customer_unique_id'])
    
    # 4. Cálculo de Total Value (Precio + Flete)
    # Rellenamos nulos en precio con 0 por si acaso (aunque el filtro delivered ayuda)
    df_clean['price'] = df_clean['price'].fillna(0)
    df_clean['freight_value'] = df_clean['freight_value'].fillna(0)
    df_clean['total_value'] = df_clean['price'] + df_clean['freight_value']

    return df_clean

def calculate_rfm_metrics(df):
    """
    Calcula las métricas Recencia, Frecuencia y Monetización por usuario único.
    
    Args:
        df (pd.DataFrame): DataFrame transaccional limpio.

    Returns:
        pd.DataFrame: Dataset con columnas R, F, M y Scores (1-5).
    """
    # Definir "Ahora" como el día siguiente a la última compra registrada
    present_date = df['order_purchase_timestamp'].max() + dt.timedelta(days=1)

    # Agrupación RFM
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (present_date - x.max()).days, # Recencia
        'order_id': 'nunique',                                               # Frecuencia
        'total_value': 'sum'                                                 # Monetización
    }).reset_index()

    rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

    # --- SCORING ---
    # Etiquetas: 5 es mejor, 1 es peor
    labels_best_high = [1, 2, 3, 4, 5] # Para F y M
    labels_best_low = [5, 4, 3, 2, 1]  # Para R (Menos días es mejor)

    # R Score
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=labels_best_low)
    
    # F Score (Uso de Rank method='first' para romper empates en distribución sesgada)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=labels_best_high)
    
    # M Score
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=labels_best_high)

    # Suma para segmentación
    rfm['RFM_Score_Sum'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)

    return rfm

def segment_customers(rfm_df):
    """
    Aplica reglas de negocio para categorizar usuarios (Gold/Silver/Bronze).

    Args:
        rfm_df (pd.DataFrame): DataFrame con scores RFM calculados.

    Returns:
        pd.DataFrame: El mismo DF con una columna nueva 'Segment'.
    """
    def _categorize(score):
        if score >= 12:
            return 'Gold'
        elif score >= 8:
            return 'Silver'
        else:
            return 'Bronze'

    rfm_df['Segment'] = rfm_df['RFM_Score_Sum'].apply(_categorize)
    return rfm_df
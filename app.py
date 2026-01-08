import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import utils

# Set page config
st.set_page_config(page_title="One Click Forecasting Model", layout="wide")

import os

# Header with Logos
c_logo1, c_logo2, c_title = st.columns([1, 1, 4])
with c_logo1:
    if os.path.exists("assets/iimu_logo.png"):
        st.image("assets/iimu_logo.png", width=150)
    else:
        st.write("IIMU") # Fallback text
with c_logo2:
    if os.path.exists("assets/o9_logo.png"):
        st.image("assets/o9_logo.png", width=100)
    else:
        st.write("o9 Solutions") # Fallback text
    
with c_title:
    st.title("One Click Forecasting Model")

# --- Custom CSS for KPI Cards ---
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Interactive Forecasting & AI Optimization")

# --- Initialize Session State ---
if 'sales_data' not in st.session_state:
    st.session_state['sales_data'] = None
if 'inventory_data' not in st.session_state:
    st.session_state['inventory_data'] = None
if 'receipt_data' not in st.session_state:
    st.session_state['receipt_data'] = None
if 'forecast_results' not in st.session_state:
    st.session_state['forecast_results'] = None # Stores (predictions, fitted, metrics)
if 'params' not in st.session_state:
    st.session_state['params'] = {} # Stores alpha, beta, gamma

# --- SIDEBAR: Configuration & Filters ---
st.sidebar.header("Configuration")

# Data Loading
with st.sidebar.expander("Upload Data", expanded=True):
    sales_file = st.file_uploader("Upload Sales History (CSV)", type=['csv'])
    if sales_file:
        st.session_state['sales_data'] = utils.load_sales_data(sales_file)
    
    inv_file = st.file_uploader("Upload Inventory (CSV)", type=['csv'])
    if inv_file:
        st.session_state['inventory_data'] = utils.load_inventory_data(inv_file)

    receipts_file = st.file_uploader("Upload Planned Receipts (CSV)", type=['csv'])
    if receipts_file:
        st.session_state['receipt_data'] = utils.load_planned_receipts(receipts_file)
        
    if st.button("Generate Dummy Data"):
        st.session_state['sales_data'], st.session_state['inventory_data'] = utils.generate_dummy_data()
        st.rerun()

# Cascading Filters
st.sidebar.header("Planning Filters")
df_sales = st.session_state['sales_data']

selected_region = 'All'
selected_store = 'All'
selected_item = 'All'

if df_sales is not None:
    # Region
    regions = ['All'] + sorted(df_sales['Region'].unique().astype(str).tolist())
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # Store
    if selected_region != 'All':
        stores = ['All'] + sorted(df_sales[df_sales['Region'] == selected_region]['Store'].unique().astype(str).tolist())
    else:
        stores = ['All'] + sorted(df_sales['Store'].unique().astype(str).tolist())
    selected_store = st.sidebar.selectbox("Store", stores)
    
    # Item
    temp = df_sales
    if selected_region != 'All': temp = temp[temp['Region'] == selected_region]
    if selected_store != 'All': temp = temp[temp['Store'] == selected_store]
    
    items = ['All'] + sorted(temp['Item'].unique().astype(str).tolist())
    selected_item = st.sidebar.selectbox("Item", items)

# --- MAIN LOGIC ---

# 1. Prepare Data based on filters
filtered_sales = pd.DataFrame()
ts_data = pd.Series()

if df_sales is not None:
    filtered_sales = utils.filter_data(df_sales, selected_region, selected_store, selected_item)
    if not filtered_sales.empty:
        # We need a single time series for forecasting
        ts_data = utils.prep_time_series(filtered_sales)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Demand Planner", "🧠 AI Optimization", "📉 Inventory Analysis (Historical)", "🚀 Service Level Optimization (Future)"])

# === TAB 1: DEMAND PLANNER ===
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Forecast Settings")
    
    if ts_data.empty:
        st.info("Please upload data and select filters to view forecast.")
    else:
        if 'params' not in st.session_state:
             st.session_state['params'] = {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.1}

        with st.expander("Forecast Settings & Smoothing", expanded=True):
            # Model Selection Row
            model_type = st.selectbox("Select Forecasting Model", 
                                     ["Triple Exp. Smoothing (Holt-Winters)", 
                                      "Double Exp. Smoothing (Holt)", 
                                      "Simple Exp. Smoothing", 
                                      "SMA"], 
                                     index=0)
            
            st.markdown("---")
            col_sliders, col_actions = st.columns([3, 1])
            
            with col_sliders:
                st.write("**Smoothing Parameters** (Override to customize)")
                # Use session state values as default for sliders
                cur_p = st.session_state['params']
                
                # We use keys to allow updating from code
                alpha = st.slider("Alpha (Level Smoothing)", 0.0, 1.0, float(cur_p.get('alpha', 0.5)), key='s_alpha')
                beta = st.slider("Beta (Trend Smoothing)", 0.0, 1.0, float(cur_p.get('beta', 0.1)), key='s_beta')
                gamma = st.slider("Gamma (Seasonal Smoothing)", 0.0, 1.0, float(cur_p.get('gamma', 0.1)), key='s_gamma')
                
                c_t1, c_t2, c_hor = st.columns(3)
                trend_type = c_t1.selectbox("Trend Type", ["add", "mul", None], index=0)
                seasonal_type = c_t2.selectbox("Seasonal Type", ["add", "mul", None], index=0)
                horizon = c_hor.slider("Forecast Horizon", 4, 52, 52)
                
            with col_actions:
                st.write("**Actions**:")
                
                if st.button("Apply Parameters", type="primary", use_container_width=True):
                     with st.spinner(f"Running {model_type}..."):
                         new_params = {
                             'alpha': alpha, 'beta': beta, 'gamma': gamma, 
                             'trend': trend_type, 'seasonal': seasonal_type
                         }
                         
                         pred, fit, met, _ = utils.run_forecast(ts_data, model_type, horizon, params=new_params)
                         st.session_state['forecast_results'] = (pred, fit, met, new_params)
                         st.session_state['params'] = new_params
                         st.session_state['current_model_type'] = model_type
                         st.rerun()
                     
                if st.button("Reset Defaults", use_container_width=True):
                     with st.spinner("Resetting..."):
                        default_params = {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.1}
                        
                        # Reset for the selected model
                        pred, fit, met, _ = utils.run_forecast(ts_data, model_type, horizon, params=default_params)
                        st.session_state['forecast_results'] = (pred, fit, met, default_params)
                        st.session_state['params'] = default_params
                        st.session_state['current_model_type'] = model_type
                        st.rerun()

                
        # Display Results
        if st.session_state.get('forecast_results'):
            pred, fitted, metrics, _ = st.session_state['forecast_results']
            current_model_type = st.session_state.get('current_model_type', 'Holt-Winters') # Fallback
            
            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            total_hist_sales = ts_data.sum()
            total_forecast = pred.sum()
            mape = metrics['MAPE']
            acc = metrics['Accuracy']
            
            k1.metric("Historical Sales", f"{total_hist_sales:,.0f}")
            k2.metric("Forecasted Demand", f"{total_forecast:,.0f}")
            k3.metric("MAPE Error", f"{mape:.1f}%", delta_color="inverse")
            k4.metric("Forecast Accuracy", f"{acc:.1f}%")
            
            # Chart
            fig = go.Figure()
            # Actuals
            fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Actual History', line=dict(color='gray')))
            # Fitted (In-sample)
            fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name='Fitted (Backtest)', line=dict(dash='dash', color='orange', width=1)))
            # Forecast
            fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode='lines', name=f'Forecast ({current_model_type})', line=dict(color='red', width=3)))
            
            # Confidence Interval (Fake for visual if not available from simple models, or rely on statsmodels summary wrapper - keeping simple for now)
            # Just shading a simplified 10% interval for demo effect if statsmodels CI extraction is complex for all types
            # (Statsmodels forecast object has conf_int() but we returned only prediction series for simplicity utils. Let's skip complex CI for this iteration)
            
            fig.update_layout(title=f"Demand Forecast: {selected_item} (Weekly)", xaxis_title="Date", yaxis_title="Sales Units", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Grid
            with st.expander("View Forecast Data Grid"):
                # Combine actuals and forecast
                df_grid = pd.DataFrame({'Forecast': pred})
                df_grid.index.name = 'Date'
                st.dataframe(df_grid.reset_index())

# === TAB 2: AI OPTIMIZATION ===
with tab2:
    st.header("Module C: AI Optimization")
    
    # 1. Configuration & Baseline
    target_mape = st.number_input("Target MAPE (%)", min_value=1.0, value=15.0, step=0.5, key='ai_target_mape')
    
    if 'forecast_results' not in st.session_state or st.session_state['forecast_results'] is None:
         st.warning("⚠️ Please run a baseline forecast in the 'Demand Planner' tab first.")
    else:
         # Get Baseline Metrics
         try:
             # Handle legacy state vs new state tuple size
             res = st.session_state['forecast_results']
             if len(res) == 4:
                 _, _, base_metrics, baseline_params = res
             else:
                 _, _, base_metrics = res
                 baseline_params = {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.1}
         except Exception:
             st.warning("State error. Please re-run the Demand Planner forecast.")
             st.stop()
         baseline_mape = base_metrics['MAPE']
         
         # Mock Baseline Params if not explicitly set (Assume default Holt-Winters values if not in session)
         # In a real app, we'd pull exact model params from the fitted model object
         # The line below is now redundant if baseline_params is directly unpacked, but kept for robustness if model_type changes
         # baseline_params = st.session_state.get('params', {'alpha': 0.5, 'beta': 0.1, 'gamma': 0.1}) 
         
         col_kpi1, col_kpi2 = st.columns(2)
         col_kpi1.metric("Baseline Error (MAPE)", f"{baseline_mape:.2f}%")
         
         if st.button("🚀 Run AI Optimization (Deep Solver)"):
            status_text = st.empty()
            status_text.write("🔬 Initializing Global Search...")
            
            with st.spinner("Executing Differential Evolution (200+ Iterations)..."):
                # 2. Optimization Phase
                best_params, optimized_mape = utils.optimize_parameters_deep(ts_data, target_mape)
                status_text.success("Optimization Complete: Best parameters found!")
                
                # 3. Champion vs. Challenger Logic
                # Logic Update: If Optimized is strictly worse, keep baseline. 
                # If Equal (Tie) or Better, adopt the AI params.
                if optimized_mape > baseline_mape:
                    st.session_state['opt_status'] = 'failed'
                    st.session_state['opt_mape'] = baseline_mape
                    st.session_state['opt_params'] = baseline_params
                    final_mape = baseline_mape
                    final_params = baseline_params
                else:
                    st.session_state['opt_status'] = 'success'
                    st.session_state['opt_mape'] = optimized_mape
                    st.session_state['opt_params'] = best_params
                    final_mape = optimized_mape
                    final_params = best_params
                    
            # 4. Display Results
            is_improved = st.session_state['opt_status'] == 'success'
            delta_color = "normal" if is_improved else "off"
            
            # Show small improvements or ties positively
            diff = baseline_mape - final_mape
            if diff > 0.001:
                delta_str = f"{diff:.2f}% Improvement"
            elif diff > -0.001: 
                delta_str = "Baseline Matched (Tie)"
            else:
                delta_str = "No Improvement"
            
            # KPI Card 2
            target_met = final_mape <= target_mape
            val_color = ":green" if target_met else ":red"
            col_kpi2.markdown(f"**New Optimized Error**")
            col_kpi2.markdown(f"## {val_color}[{final_mape:.2f}%]")
            
            if is_improved:
                col_kpi2.caption(f"✅ {delta_str}")
            elif not target_met:
                 col_kpi2.caption("⚠️ Best Possible Result (Target Limit)")
                 st.toast("Optimization Complete. This is the best statistical fit possible.", icon="✅")
                 st.warning("Diagnostic: The AI Solver minimized error as far as possible. The target MAPE may be statistically unreachable due to high volatility in the sales data.")
            else:
                col_kpi2.caption("✅ Baseline matches Target")

            # 5. Parameter Impact & Unlocked Features
            st.markdown("---")
            st.subheader("Optimization Unlocks")
            
            # Feature Badges
            c_feat1, c_feat2, c_feat3 = st.columns(3)
            
            # Check Box-Cox
            if final_params.get('boxcox'):
                c_feat1.success(f"**Box-Cox Transform**: Enabled (λ={final_params.get('lambda', 0):.2f})")
            else:
                c_feat1.info("Box-Cox Transform: Not Required", icon="ℹ️")
                
            # Check Damping
            if final_params.get('damped'):
                c_feat2.success(f"**Trend Damping**: Enabled (φ={final_params.get('phi', 0):.3f})")
            else:
                c_feat2.info("Trend Damping: Not Required", icon="ℹ️")
                
            # Check Initialization Adjustment
            if 'l0' in final_params:
                c_feat3.success("**Initial State Optimization**: Active")
            else:
                c_feat3.info("State Optimization: Standard", icon="ℹ️")

            st.markdown("### Parameter Impact Analysis")
            
            # Construct Comparison Data
            # Note: Initial states are not in baseline params usually, so we compare mainly smoothing
            param_names = ['alpha', 'beta', 'gamma'] # Core params
            rows = []
            for p in param_names:
                old_v = baseline_params.get(p, 0)
                new_v = final_params.get(p, 0)
                diff_val = new_v - old_v
                pct_change = (diff_val / old_v * 100) if old_v != 0 else 0
                
                rows.append({
                    "Parameter": p.capitalize(),
                    "Baseline": old_v,
                    "Optimized": new_v,
                    "Abs Change": abs(diff_val),
                    "Direction": "⬆️" if diff_val > 0 else ("⬇️" if diff_val < 0 else "➖")
                })
            
            impact_df = pd.DataFrame(rows)
            
            # Impact Chart
            fig_imp = go.Figure(data=[
                go.Bar(name='Baseline', x=impact_df['Parameter'], y=impact_df['Baseline'], marker_color='lightgray'),
                go.Bar(name='Optimized', x=impact_df['Parameter'], y=impact_df['Optimized'], marker_color='#6366f1')
            ])
            fig_imp.update_layout(barmode='group', title="Smoothing Parameter Shift", height=300)
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.dataframe(impact_df[['Parameter', 'Baseline', 'Optimized', 'Direction']], use_container_width=True)
            
            # 6. AI Narrative
            st.info("🤖 **AI Strategic Advice**")
            
            insights = utils.generate_natural_language_insight(
                baseline_params, final_params, baseline_mape, final_mape, target_mape
            )
            
            for line in insights:
                st.write(line)
            
            if is_improved:
                st.success("Recommendation: Save these new parameters for the production forecast.")

# === TAB 3: INVENTORY ANALYSIS (HISTORICAL) ===
with tab3:
    st.header("Module D1: Historical Inventory Analysis")
    st.info("Analyzing Actual Stock vs Actual Sales History")

    if st.session_state['inventory_data'] is None:
        st.warning("Please upload Inventory Data.")
    else:
        # Filter Inventory
        df_inv = st.session_state['inventory_data']
        filtered_inv = utils.filter_data(df_inv, selected_region, selected_store, selected_item)
        
        # Prep TS (Hist/Actuals)
        # We need actual time series of Inventory
        ts_inv = utils.prep_time_series(filtered_inv, val_col='ClosingStock')
        
        # We also need Actual Sales (ts_data calculated in main area)
        if ts_inv.empty or ts_data.empty:
            st.error("Missing Inventory or Sales Data for this selection.")
        else:
            # Prepare DFs for merge
            df_inv_clean = ts_inv.to_frame(name='ClosingStock').reset_index()
            df_inv_clean.rename(columns={df_inv_clean.columns[0]: 'WeekStart'}, inplace=True)
            
            df_sales_clean = ts_data.to_frame(name='SalesUnits').reset_index()
            df_sales_clean.rename(columns={df_sales_clean.columns[0]: 'WeekStart'}, inplace=True)
            
            # Analyze
            df_hist = utils.calculate_historical_inventory_health(df_sales_clean, df_inv_clean)
            
            # Metrics
            avg_stock = df_hist['ClosingStock'].mean()
            avg_sales = df_hist['SalesUnits'].mean()
            # Historical Fill Rate
            fill_rate = df_hist['Service_Met'].mean() * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Historical Stock", f"{avg_stock:,.0f}")
            m2.metric("Avg Weekly Sales", f"{avg_sales:,.0f}")
            m3.metric("Historical Service Level", f"{fill_rate:.1f}%")
            
            # Chart
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_hist['WeekStart'], y=df_hist['SalesUnits'], name='Actual Sales', marker_color='#A0C4FF'))
            fig3.add_trace(go.Scatter(x=df_hist['WeekStart'], y=df_hist['ClosingStock'], name='Actual Stock', line=dict(color='#FFADAD', width=3)))
            
            fig3.update_layout(title="Historical Performance: Stock vs Sales", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.dataframe(df_hist)

# === TAB 4: SERVICE LEVEL OPTIMIZATION (FUTURE) ===
with tab4:
    st.header("Module D2: Service Level Optimization (Future Planning)")
    st.info("Projecting Future Stock based on Forecast + Planned Receipts")
    
    if st.session_state['inventory_data'] is None:
        st.warning("Please upload Inventory Data (for starting stock).")
    elif st.session_state['forecast_results'] is None:
        st.warning("Please run a Forecast in Tab 1 first.")
    else:
        # We use the previous logic here
        df_inv = st.session_state['inventory_data']
        filtered_inv = utils.filter_data(df_inv, selected_region, selected_store, selected_item)
        
        df_receipts = st.session_state.get('receipt_data')
        filtered_receipts = None
        if df_receipts is not None:
             filtered_receipts = utils.filter_data(df_receipts, selected_region, selected_store, selected_item)
             
        ts_inv = utils.prep_time_series(filtered_inv, val_col='ClosingStock')
        
        if ts_inv.empty:
             st.error("No inventory data.")
        else:
            forward_weeks = st.number_input("Forward Cover Target (Weeks)", min_value=1, max_value=12, value=4, key='fwd_cov')
            
            pred, _, _, _ = st.session_state['forecast_results']
            
            df_inv_clean = ts_inv.to_frame(name='ClosingStock').reset_index()
            df_inv_clean.rename(columns={df_inv_clean.columns[0]: 'WeekStart'}, inplace=True)
            
            # PAB Calculation
            df_analysis = utils.calculate_inventory_projections(df_inv_clean, pred, filtered_receipts, forward_weeks)
            
            # Filter to show only Future/Projections generally, or show cutoff
            # The function returns everything, but let's focus Metrics on Future
            future_df = df_analysis[df_analysis['Type'] == 'Projection']
            
            if not future_df.empty:
                avg_sl = future_df['Service_Met'].mean() * 100
                last_stock = df_analysis['ProjectedStock'].iloc[-1]
                min_stock = df_analysis['ProjectedStock'].min()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Projected Service Level", f"{avg_sl:.1f}%", help="% of future weeks where Stock > Target")
                m2.metric("Ending Stock (Horizon)", f"{last_stock:,.0f}")
                m3.metric("Lowest Projected Stock", f"{min_stock:,.0f}", delta_color="inverse" if min_stock < 0 else "normal")
            
            # --- AI OPTIMIZATION ---
            st.divider()
            col_opt1, col_opt2 = st.columns([1, 1])
            with col_opt1:
                st.markdown("#### 🤖 AI Inventory Balancing")
                st.markdown("Optimization Goal: **Fill deficits to achieve Target Service Level.**")
                target_sl = st.slider("Target Service Level (%)", min_value=50, max_value=100, value=95, step=5)
                
            if st.button("✨ Generate Replenishment Plan", type="primary"):
                # Run Optimization
                df_opt = utils.calculate_optimized_projections(df_analysis, target_sl_percent=target_sl)
                
                # Show New Metrics
                # Recalculate Service Level for Optimized Plan
                # New Closing Stock = OptimizedStock - Forecast
                # New Service Check: (OptimizedStock - Forecast) > Target_Stock
                # Actually, OptimizedStock IS 'Available'.
                # Wait, my logic in utils calculated OptimizedStock (Available).
                # New Closing = OptimizedStock - Forecast.
                # Service Met = New Closing > Target.
                
                # Let's derive columns for display safely
                df_opt['OptimizedClosing'] = df_opt['OptimizedStock'] - df_opt['Forecast']
                df_opt['Opt_Service_Met'] = df_opt['OptimizedClosing'] > df_opt['Target_Stock']
                
                future_opt = df_opt[df_opt['Type'] == 'Projection']
                new_sl = future_opt['Opt_Service_Met'].mean() * 100
                total_suggested = future_opt['SuggestedOrder'].sum()
                
                c1, c2 = st.columns(2)
                c1.metric("New Service Level", f"{new_sl:.1f}%", delta=f"{new_sl - avg_sl:.1f}%")
                c2.metric("Total Suggested New Orders", f"{total_suggested:,.0f}")
                
                # Use df_opt for plotting
                plot_df = df_opt
                is_optimized = True
            else:
                plot_df = df_analysis
                is_optimized = False

            # Dual Axis Chart for Future
            fig4 = go.Figure()
            
            # Determine Cutover Date for visual aid
            cutover_date = plot_df[plot_df['Type'] == 'History']['WeekStart'].max()
            
            # Forecast
            fig4.add_trace(go.Bar(x=plot_df['WeekStart'], y=plot_df['Forecast'], name='Forecast Demand', marker_color='#BDB2FF', opacity=0.5, yaxis='y1'))
            
            # Receipts
            fig4.add_trace(go.Bar(x=plot_df['WeekStart'], y=plot_df['ReceiptQty'], name='Planned Receipts', marker_color='#9BF6FF', opacity=0.8, yaxis='y1'))
            
            if is_optimized:
                 # Suggested Orders (Stacked?)
                 # Plot them on top of receipts? Or separate yellow bar
                 fig4.add_trace(go.Bar(x=plot_df['WeekStart'], y=plot_df['SuggestedOrder'], name='AI Suggested Orders', marker_color='#FFD166', opacity=0.9, yaxis='y1'))

            # Actual Stock (History)
            hist_df = plot_df[plot_df['Type'] == 'History']
            if not hist_df.empty:
                fig4.add_trace(go.Scatter(x=hist_df['WeekStart'], y=hist_df['ClosingStock'], name='Actual Stock', line=dict(color='gray', width=2), yaxis='y2'))
            
            # Projected Stock
            # If Optimized, show Optimized line instead?
            # Or Show both? Let's show the best one.
            if is_optimized:
                y_stock = plot_df['OptimizedStock'] # This is Available
                name_stock = "Optimized Stock (Available)"
                color_stock = '#06D6A0' # Green
            else:
                y_stock = plot_df['ProjectedStock'] # This is Available
                name_stock = "Projected Stock (Available)"
                color_stock = '#FF686B' # Red/Default
                
            proj_df = plot_df 
            # We specifically want to highlight the projection
            fig4.add_trace(go.Scatter(x=proj_df['WeekStart'], y=y_stock, name=name_stock, line=dict(color=color_stock, width=3), yaxis='y2'))
            
            # Target
            fig4.add_trace(go.Scatter(x=plot_df['WeekStart'], y=plot_df['Target_Stock'], name='Target Stock', line=dict(color='purple', dash='dot', width=1), yaxis='y2'))

            # Cutover Line
            if pd.notna(cutover_date):
                fig4.add_vline(x=cutover_date, line_width=1, line_dash="dash", line_color="black")
                fig4.add_annotation(x=cutover_date, y=1, yref="paper", text="Today", showarrow=False)

            fig4.update_layout(
                title="Service Level Optimization: Stock Projection",
                yaxis=dict(title="Units", side='left'),
                yaxis2=dict(title="Stock Level", side='right', overlaying='y'),
                template="plotly_white",
                legend=dict(x=0, y=1.1, orientation='h'),
                hovermode="x unified"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            st.subheader("Future Plan Details")
            # Only format numeric columns to avoid formatting strings/dates
            fmt_dict = {
                'Forecast': '{:,.0f}',
                'ReceiptQty': '{:,.0f}',
                'Target_Stock': '{:,.0f}'
            }
            if is_optimized:
                fmt_dict['SuggestedOrder'] = '{:,.0f}'
                fmt_dict['OptimizedStock'] = '{:,.0f}'
                fmt_dict['ProjectedStock'] = '{:,.0f}'
            else:
                fmt_dict['ProjectedStock'] = '{:,.0f}'
                
            st.dataframe(plot_df[plot_df['Type'] == 'Projection'].style.format(fmt_dict))

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import datetime as dt
import pandas as pd

def Q3_plot1(otm_options: pd.DataFrame) -> plt.Figure:
    
    moneyness_col='moneyness'
    options_jan17 = otm_options[otm_options['date'] == dt.date(2020,1,17)]
    options_mar20 = otm_options[otm_options['date'] == dt.date(2020,3,20)]

    maturities_jan17 = [dt.date(2020,2,14), dt.date(2020,7,17)]
    maturities_mar20 = [dt.date(2020,4,17), dt.date(2020,10,16)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Jan 17 options
    ax = axes[0]
    for exdate in maturities_jan17:
        df = options_jan17[options_jan17['exdate'] == exdate]
        ax.scatter(df[moneyness_col], df['self_calc_imp_vol_BMS'], s=20, alpha=0.6, label=f'Self Calc IV exp:{exdate}')
        df_sorted = df.sort_values(moneyness_col)
        ax.plot(df_sorted[moneyness_col], df_sorted['implied_vol_bms'], label=f'Provider IV exp:{exdate}')
    ax.set_title('Options quoted on Jan 17, 2020')
    ax.set_xlabel('Moneyness (K / S_t)')
    ax.set_ylabel('Implied Volatility')
    ax.grid(True)
    ax.legend()
    # Add horizontal arrow below x-axis for left subplot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate('', xy=(1.4, -0.16), xytext=(1.01, -0.16),
                xycoords=trans,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(1.2, -0.18, 'OTM Calls', transform=trans,
            ha='center', va='top', fontsize=10, color='red')
    ax.annotate('', xy=(0.6, -0.16), xytext=(0.99, -0.16),
                xycoords=trans,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(0.8, -0.18, 'OTM Puts', transform=trans,
            ha='center', va='top', fontsize=10, color='red')

    # Right panel: Mar 20 options
    ax = axes[1]
    for exdate in maturities_mar20:
        df = options_mar20[options_mar20['exdate'] == exdate]
        ax.scatter(df[moneyness_col], df['self_calc_imp_vol_BMS'], s=20, alpha=0.6, label=f'Self Calc IV exp:{exdate}')
        df_sorted = df.sort_values(moneyness_col)
        ax.plot(df_sorted[moneyness_col], df_sorted['implied_vol_bms'], label=f'Provider IV exp:{exdate}')
    ax.set_title('Options quoted on Mar 20, 2020')
    ax.set_xlabel('Moneyness (K / S_t)')
    ax.set_ylabel('Implied Volatility')
    ax.grid(True)
    ax.legend()
    # Add horizontal arrow below x-axis for left subplot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate('', xy=(1.4, -0.16), xytext=(1.01, -0.16),
                xycoords=trans,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(1.2, -0.18, 'OTM Calls', transform=trans,
            ha='center', va='top', fontsize=10, color='red')
    ax.annotate('', xy=(0.6, -0.16), xytext=(0.99, -0.16),
                xycoords=trans,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(0.8, -0.18, 'OTM Puts', transform=trans,
            ha='center', va='top', fontsize=10, color='red')

    plt.suptitle('Implied Volatility Smiles for OTM Options')
    return fig

def Q3_plot2(otm_options: pd.DataFrame) -> plt.Figure:
    
    moneyness_col='moneyness_adjusted'
    options_jan17 = otm_options[otm_options['date'] == dt.date(2020,1,17)]
    options_mar20 = otm_options[otm_options['date'] == dt.date(2020,3,20)]

    maturities_jan17 = [dt.date(2020,2,14), dt.date(2020,7,17)]
    maturities_mar20 = [dt.date(2020,4,17), dt.date(2020,10,16)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Jan 17 options
    ax = axes[0]
    for exdate in maturities_jan17:
        df = options_jan17[options_jan17['exdate'] == exdate]
        ax.scatter(df[moneyness_col], df['self_calc_imp_vol_BMS'], s=20, alpha=0.6, label=f'Self Calc IV exp:{exdate}')
        df_sorted = df.sort_values(moneyness_col)
        ax.plot(df_sorted[moneyness_col], df_sorted['implied_vol_bms'], label=f'Provider IV exp:{exdate}')
    ax.set_title('Options quoted on Jan 17, 2020')
    ax.set_xlabel(r'Standardized Moneyness $= \ln(K/S) / (\sigma\sqrt{\tau})$')
    ax.set_ylabel('Implied Volatility')
    ax.grid(True)
    ax.legend()

    # Right panel: Mar 20 options
    ax = axes[1]
    for exdate in maturities_mar20:
        df = options_mar20[options_mar20['exdate'] == exdate]
        ax.scatter(df[moneyness_col], df['self_calc_imp_vol_BMS'], s=20, alpha=0.6, label=f'Self Calc IV exp:{exdate}')
        df_sorted = df.sort_values(moneyness_col)
        ax.plot(df_sorted[moneyness_col], df_sorted['implied_vol_bms'], label=f'Provider IV exp:{exdate}')
    ax.set_title('Options quoted on Mar 20, 2020')
    ax.set_xlabel(r'Standardized Moneyness $= \ln(K/S) / (\sigma\sqrt{\tau})$')
    ax.set_ylabel('Implied Volatility')
    ax.grid(True)
    ax.legend()

    plt.suptitle('Implied Volatility Smiles for OTM Options with Adjusted Moneyness')
    return fig


def Q3_plot(otm_options: pd.DataFrame) -> plt.Figure:
    
    # Filter options by quote date
    options_jan17 = otm_options[otm_options['date'] == dt.date(2020, 1, 17)]
    options_mar20 = otm_options[otm_options['date'] == dt.date(2020, 3, 20)]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: January 17, 2020
    ax = axes[0]
    jan_puts = options_jan17[options_jan17['cp_flag'] == 'P']
    jan_calls = options_jan17[options_jan17['cp_flag'] == 'C']

    ax.scatter(jan_puts['moneyness'], 
            100 * (jan_puts['impl_volatility'] / jan_puts['implied_vol_bms'] - 1),
            alpha=0.6, s=20, label='Puts', color='red')

    ax.scatter(jan_calls['moneyness'], 
            100 * (jan_calls['impl_volatility'] / jan_calls['implied_vol_bms'] - 1),
            alpha=0.6, s=20, label='Calls', color='blue')

    ax.set_xlabel('Moneyness (K / S_t)')
    ax.set_ylabel('100 × (impl_volatility/implied_vol_bms - 1) %')
    ax.set_title('Quote Date: 2020-01-17')
    ax.grid(True)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Right subplot: March 20, 2020
    ax = axes[1]
    mar_puts = options_mar20[options_mar20['cp_flag'] == 'P']
    mar_calls = options_mar20[options_mar20['cp_flag'] == 'C']

    ax.scatter(mar_puts['moneyness'], 
            100 * (mar_puts['impl_volatility'] / mar_puts['implied_vol_bms'] - 1),
            alpha=0.6, s=20, label='Puts', color='red')

    ax.scatter(mar_calls['moneyness'], 
            100 * (mar_calls['impl_volatility'] / mar_calls['implied_vol_bms'] - 1),
            alpha=0.6, s=20, label='Calls', color='blue')

    ax.set_xlabel('Moneyness (K / S_t)')
    ax.set_ylabel('100 × (impl_volatility/implied_vol_bms - 1) %')
    ax.set_title('Quote Date: 2020-03-20')
    ax.grid(True)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.suptitle('Percentage Difference Between Provider and BMS Implied Volatilities for OTM Options', 
                fontsize=12, y=1.02)
    
    return fig


def Q4_plot(otm_options: pd.DataFrame) -> plt.Figure:
    
    # Filter options by quote date
    options_jan17 = otm_options[otm_options['date'] == dt.date(2020, 1, 17)]
    options_mar20 = otm_options[otm_options['date'] == dt.date(2020, 3, 20)]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: January 17, 2020
    ax = axes[0]
    jan_puts = options_jan17[options_jan17['cp_flag'] == 'P']
    jan_calls = options_jan17[options_jan17['cp_flag'] == 'C']

    ax.scatter(jan_puts['moneyness'], 
            100 * (jan_puts['impl_volatility'] / jan_puts['crr_implied_vol'] - 1),
            alpha=0.6, s=20, label='Puts', color='red')

    ax.scatter(jan_calls['moneyness'], 
            100 * (jan_calls['impl_volatility'] / jan_calls['crr_implied_vol'] - 1),
            alpha=0.6, s=20, label='Calls', color='blue')

    ax.set_xlabel('Moneyness (K / S_t)')
    ax.set_ylabel('100 × (impl_volatility/crr_implied_vol - 1) %')
    ax.set_title('Quote Date: 2020-01-17')
    ax.grid(True)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Right subplot: March 20, 2020
    ax = axes[1]
    mar_puts = options_mar20[options_mar20['cp_flag'] == 'P']
    mar_calls = options_mar20[options_mar20['cp_flag'] == 'C']

    ax.scatter(mar_puts['moneyness'], 
            100 * (mar_puts['impl_volatility'] / mar_puts['crr_implied_vol'] - 1),
            alpha=0.6, s=20, label='Puts', color='red')

    ax.scatter(mar_calls['moneyness'], 
            100 * (mar_calls['impl_volatility'] / mar_calls['crr_implied_vol'] - 1),
            alpha=0.6, s=20, label='Calls', color='blue')

    ax.set_xlabel('Moneyness (K / S_t)')
    ax.set_ylabel('100 × (impl_volatility/crr_implied_vol - 1) %')
    ax.set_title('Quote Date: 2020-03-20')
    ax.grid(True)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.suptitle('Percentage Difference Between Provider and CRR Implied Volatilities for OTM Options', 
                fontsize=12, y=1.02)
    
    return fig


def Q6_plot(options: pd.DataFrame) -> plt.Figure:
    
        # Filter by dates
        options_jan17 = options[options['date'] == dt.date(2020, 1, 17)]
        options_mar20 = options[options['date'] == dt.date(2020, 3, 20)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # ----------- Jan 17 subplot -----------
        ax = axes[0]
        calls_jan = options_jan17[options_jan17['is_call']]
        puts_jan = options_jan17[~options_jan17['is_call']]
        ax.scatter(calls_jan['moneyness'], calls_jan['early_exercise_premium'],
                c='blue', alpha=0.6, s=20, label='Calls')
        ax.scatter(puts_jan['moneyness'], puts_jan['early_exercise_premium'],
                c='red', alpha=0.6, s=20, label='Puts')
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Early Exercise Premium ($)')
        ax.set_title('Early Exercise Premium - 2020-01-17')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.grid(True)
        # Draw horizontal arrow for OTM puts and ITM calls
        ax.annotate('', xy=(0.7, 7), xytext=(0.98, 7),
        arrowprops=dict(arrowstyle='-|>', color='green', linewidth=3))

        ax.text(0.85, 7.15, 'OTM puts', ha='center', va='bottom', fontsize=14, color='red')
        ax.text(0.85, 6.85, 'ITM calls', ha='center', va='top', fontsize=14, color='blue')

        # Draw horizontal arrow for ITM puts and OTM calls
        ax.annotate('', xy=(1.3, 7), xytext=(1.02, 7),
        arrowprops=dict(arrowstyle='-|>', color='green', linewidth=3))

        ax.text(1.15, 7.15, 'ITM puts', ha='center', va='bottom', fontsize=14, color='red')
        ax.text(1.15, 6.85, 'OTM calls', ha='center', va='top', fontsize=14, color='blue')

        # ----------- Mar 20 subplot -----------
        ax = axes[1]
        calls_mar = options_mar20[options_mar20['is_call']]
        puts_mar = options_mar20[~options_mar20['is_call']]
        ax.scatter(calls_mar['moneyness'], calls_mar['early_exercise_premium'],
                c='blue', alpha=0.6, s=20, label='Calls')
        ax.scatter(puts_mar['moneyness'], puts_mar['early_exercise_premium'],
                c='red', alpha=0.6, s=20, label='Puts')
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Early Exercise Premium ($)')
        ax.set_title('Early Exercise Premium - 2020-03-20')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.grid(True)
        # Draw horizontal arrow for OTM puts and ITM calls
        ax.annotate('', xy=(0.6, 3), xytext=(0.98, 3),
        arrowprops=dict(arrowstyle='-|>', color='green', linewidth=3))

        ax.text(0.8, 3.05, 'OTM puts', ha='center', va='bottom', fontsize=14, color='red')
        ax.text(0.8, 2.9, 'ITM calls', ha='center', va='top', fontsize=14, color='blue')

        # Draw horizontal arrow for ITM puts and OTM calls
        ax.annotate('', xy=(1.4, 3), xytext=(1.02, 3),
        arrowprops=dict(arrowstyle='-|>', color='green', linewidth=3))

        ax.text(1.2, 3.05, 'ITM puts', ha='center', va='bottom', fontsize=14, color='red')
        ax.text(1.2, 2.9, 'OTM calls', ha='center', va='top', fontsize=14, color='blue')

        return fig
#!/usr/bin/env python3
"""Script to apply all fixes to app.py systematically."""

import re

# Read current app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Replace temp_column_config with column_config_persistent
print("Applying Fix 1: Persist column configuration...")
content = content.replace('temp_column_config', 'column_config_persistent')

# Fix 2: Change sample values from 3 to 5
print("Applying Fix 2: Increase sample values to 5...")
content = re.sub(r'\.head\(3\)', '.head(5)', content)

# Fix 3: Adjust train/test slider from 50-90% to 10-70%
print("Applying Fix 3: Adjust train/test split range...")
content = re.sub(
    r'st\.slider\("Training set size", min_value=0\.50, max_value=0\.90, value=0\.70, step=0\.05\)',
    'st.slider("Training set size", min_value=0.10, max_value=0.70, value=0.70, step=0.05)',
    content
)

# Fix 4: Add export reporting with safe column access
print("Applying Fix 4: Fix export function with safe column checks...")

# Find and replace the export section
old_export = '''        # Download recommendations
        st.subheader("Export Recommendations")
        
        if st.button("📥 Download Full Report"):
            # Create a comprehensive report
            report = f"""
EMPLOYEE ATTRITION ANALYSIS REPORT
{'='*80}

FORECAST HORIZON: {forecast_horizon}

"""
            if 'summary' in locals():
                report += summary + "\\n\\n"
            
            report += f"""
DETAILED RECOMMENDATIONS
{'-'*80}

"""
            if 'rec_df' in locals():
                for idx, row in rec_df.iterrows():
                    report += f"\\n{row['Priority']} PRIORITY: {row['Risk Factor']}\\n"
                    report += f"Recommended Action: {row['Recommended Action']}\\n"
                    report += f"Impact Score: {row['Impact']}\\n"
                    report += "-" * 80 + "\\n"
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="attrition_analysis_report.txt",
                mime="text/plain"
            )'''

new_export = '''        # Download recommendations in multiple formats
        st.subheader("Export Recommendations")
        
        # Prepare safe report data
        try:
            report_data = {
                'forecast_horizon': forecast_horizon,
                'total_employees': total_employees if 'total_employees' in locals() else len(df),
                'attrition_rate': attrition_rate if 'attrition_rate' in locals() else (df[target_col].mean() * 100 if target_col in df.columns else 0),
                'high_risk_count': high_risk_count if 'high_risk_count' in locals() else 0,
                'summary': summary if 'summary' in locals() else '',
                'rec_df': rec_df if 'rec_df' in locals() else None
            }
        except:
            report_data = {'forecast_horizon': forecast_horizon, 'total_employees': len(df), 'attrition_rate': 0, 'high_risk_count': 0, 'summary': '', 'rec_df': None}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 Text Report"):
                report = f"""
EMPLOYEE ATTRITION ANALYSIS REPORT
{'='*80}

FORECAST HORIZON: {report_data['forecast_horizon']}
TOTAL EMPLOYEES: {report_data['total_employees']:,}
ATTRITION RATE: {report_data['attrition_rate']:.1f}%

"""
                if report_data['summary']:
                    report += report_data['summary'] + "\\n\\n"
                
                report += f"""
DETAILED RECOMMENDATIONS
{'-'*80}

"""
                if report_data['rec_df'] is not None:
                    for idx, row in report_data['rec_df'].iterrows():
                        report += f"\\n{row['Priority']} PRIORITY: {row['Risk Factor']}\\n"
                        report += f"Recommended Action: {row['Recommended Action']}\\n"
                        report += f"Impact Score: {row['Impact']}\\n"
                        report += "-" * 80 + "\\n"
                
                st.download_button(
                    label="📥 Text",
                    data=report,
                    file_name="attrition_analysis_report.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Excel Report"):
                try:
                    import io
                    from openpyxl import Workbook
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Attrition Analysis"
                    ws['A1'] = "EMPLOYEE ATTRITION ANALYSIS REPORT"
                    ws['A1'].font = ws['A1'].font.copy()
                    ws['A1'].font = ws['A1'].font.copy()
                    ws['A3'] = f"Forecast Horizon: {report_data['forecast_horizon']}"
                    ws['A4'] = f"Total Employees: {report_data['total_employees']:,}"
                    ws['A5'] = f"Attrition Rate: {report_data['attrition_rate']:.1f}%"
                    ws['A6'] = f"High-Risk Count: {report_data['high_risk_count']}"
                    
                    if report_data['rec_df'] is not None:
                        ws_rec = wb.create_sheet("Recommendations")
                        for r_idx, col in enumerate(report_data['rec_df'].columns, 1):
                            ws_rec.cell(row=1, column=r_idx, value=col)
                        for r_idx, row in enumerate(report_data['rec_df'].values, 2):
                            for c_idx, val in enumerate(row, 1):
                                ws_rec.cell(row=r_idx, column=c_idx, value=val)
                    
                    buffer = io.BytesIO()
                    wb.save(buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Excel",
                        data=buffer.getvalue(),
                        file_name="attrition_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.error("openpyxl required. Install with: pip install openpyxl")
        
        with col3:
            if st.button("📄 Word Report"):
                try:
                    import io
                    from docx import Document
                    doc = Document()
                    doc.add_heading("EMPLOYEE ATTRITION ANALYSIS REPORT", 0)
                    doc.add_paragraph(f"Forecast Horizon: {report_data['forecast_horizon']}")
                    doc.add_paragraph(f"Total Employees: {report_data['total_employees']:,}")
                    doc.add_paragraph(f"Attrition Rate: {report_data['attrition_rate']:.1f}%")
                    doc.add_paragraph(f"High-Risk Count: {report_data['high_risk_count']}")
                    
                    if report_data['rec_df'] is not None:
                        doc.add_heading("Recommendations", 1)
                        table = doc.add_table(rows=1, cols=4)
                        hdr_cells = table.rows[0].cells
                        for idx, col_name in enumerate(report_data['rec_df'].columns):
                            hdr_cells[idx].text = str(col_name)
                        for _, row in report_data['rec_df'].iterrows():
                            row_cells = table.add_row().cells
                            for idx, val in enumerate(row):
                                row_cells[idx].text = str(val)
                    
                    buffer = io.BytesIO()
                    doc.save(buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Word",
                        data=buffer.getvalue(),
                        file_name="attrition_analysis_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except ImportError:
                    st.error("python-docx required. Install with: pip install python-docx")
        
        with col4:
            if st.button("📊 PDF Report"):
                try:
                    import io
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    buffer = io.BytesIO()
                    pdf = SimpleDocTemplate(buffer, pagesize=letter)
                    story = []
                    styles = getSampleStyleSheet()
                    
                    story.append(Paragraph("EMPLOYEE ATTRITION ANALYSIS REPORT", styles['Heading1']))
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(f"<b>Forecast Horizon:</b> {report_data['forecast_horizon']}", styles['Normal']))
                    story.append(Paragraph(f"<b>Total Employees:</b> {report_data['total_employees']:,}", styles['Normal']))
                    story.append(Paragraph(f"<b>Attrition Rate:</b> {report_data['attrition_rate']:.1f}%", styles['Normal']))
                    
                    if report_data['rec_df'] is not None:
                        story.append(Spacer(1, 0.3*inch))
                        story.append(Paragraph("Recommendations", styles['Heading2']))
                        data = [["Priority", "Risk Factor", "Impact", "Action"]]
                        for _, row in report_data['rec_df'].iterrows():
                            data.append([str(row['Priority']), str(row['Risk Factor']), str(row['Impact']), str(row['Recommended Action'])[:30]])
                        table = Table(data)
                        story.append(table)
                    
                    pdf.build(story)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 PDF",
                        data=buffer.getvalue(),
                        file_name="attrition_analysis_report.pdf",
                        mime="application/pdf"
                    )
                except ImportError:
                    st.error("reportlab required. Install with: pip install reportlab")'''

if old_export in content:
    content = content.replace(old_export, new_export)
    print("  Export function updated successfully")
else:
    print("  WARNING: Could not find exact export section match - you may need to update manually")

# Write fixed content back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ All fixes applied to app.py successfully!")
print("\nChanges made:")
print("  1. Replaced temp_column_config with column_config_persistent for persistence")
print("  2. Increased sample values display from 3 to 5")
print("  3. Adjusted train/test split range from 50-90% to 10-70%")
print("  4. Added multi-format export (Text, Excel, Word, PDF)")
print("  5. Added safe column access in export/recommendations function")

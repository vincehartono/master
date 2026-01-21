import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class RoulettePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Roulette Predictor - Last 6 Draws Analysis")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")
        
        # Data storage
        self.draws = []
        self.history_file = "roulette_history.json"
        self.load_history()
        
        # Color mapping
        self.red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
        self.black_numbers = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
        
        self.setup_ui()
        self.update_display()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="🎰 ROULETTE PREDICTOR 🎰", font=("Arial", 20, "bold"), 
                               bg="#2c3e50", fg="#ecf0f1")
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== INPUT SECTION =====
        input_frame = ttk.LabelFrame(main_frame, text="Input New Draw", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        input_label = tk.Label(input_frame, text="Enter number (0-36):", font=("Arial", 11), bg="#2c3e50", fg="#ecf0f1")
        input_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.input_entry = ttk.Entry(input_frame, width=10, font=("Arial", 11))
        self.input_entry.grid(row=0, column=1, padx=5)
        self.input_entry.bind("<Return>", lambda e: self.add_draw())
        
        add_btn = ttk.Button(input_frame, text="Add Draw", command=self.add_draw)
        add_btn.grid(row=0, column=2, padx=5)
        
        clear_btn = ttk.Button(input_frame, text="Clear History", command=self.clear_history)
        clear_btn.grid(row=0, column=3, padx=5)
        
        # ===== LAST 6 DRAWS SECTION =====
        history_frame = ttk.LabelFrame(main_frame, text="Last 6 Draws", padding=10)
        history_frame.pack(fill=tk.X, pady=5)
        
        self.history_display = tk.Label(history_frame, text="", font=("Arial", 12, "bold"), 
                                        bg="#2c3e50", fg="#f39c12", justify=tk.LEFT)
        self.history_display.pack(fill=tk.X)
        
        # ===== ANALYSIS SECTION =====
        analysis_frame = ttk.LabelFrame(main_frame, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create frames for each graph
        self.color_frame = ttk.Frame(analysis_frame)
        self.color_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.parity_frame = ttk.Frame(analysis_frame)
        self.parity_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.size_frame = ttk.Frame(analysis_frame)
        self.size_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== RECOMMENDATION SECTION =====
        rec_frame = ttk.LabelFrame(main_frame, text="🎯 RECOMMENDATION", padding=10)
        rec_frame.pack(fill=tk.X, pady=5)
        
        self.recommendation = tk.Label(rec_frame, text="", font=("Arial", 12, "bold"), 
                                       bg="#2c3e50", fg="#f1c40f", justify=tk.LEFT, wraplength=800)
        self.recommendation.pack(fill=tk.X)
    
    def add_draw(self):
        try:
            num = int(self.input_entry.get())
            if num < 0 or num > 36:
                messagebox.showerror("Invalid Input", "Please enter a number between 0 and 36")
                return
            
            self.draws.append(num)
            if len(self.draws) > 6:
                self.draws.pop(0)
            
            self.save_history()
            self.update_display()
            self.input_entry.delete(0, tk.END)
            self.input_entry.focus()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")
    
    def draw_color_graph(self):
        """Draw red vs black pie chart"""
        red_count = sum(1 for d in self.draws if d in self.red_numbers)
        black_count = sum(1 for d in self.draws if d in self.black_numbers)
        
        # Clear previous canvas
        for widget in self.color_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(4, 2.5), dpi=80, facecolor="#2c3e50")
        ax = fig.add_subplot(111, facecolor="#2c3e50")
        
        sizes = [red_count, black_count]
        labels = [f"RED\n({red_count}/6)", f"BLACK\n({black_count}/6)"]
        colors = ["#e74c3c", "#000000"]
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                           explode=explode, startangle=90, textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        canvas = FigureCanvasTkAgg(fig, master=self.color_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def draw_parity_graph(self):
        """Draw odd vs even pie chart"""
        odd_count = sum(1 for d in self.draws if d % 2 == 1)
        even_count = sum(1 for d in self.draws if d % 2 == 0)
        
        # Clear previous canvas
        for widget in self.parity_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(4, 2.5), dpi=80, facecolor="#2c3e50")
        ax = fig.add_subplot(111, facecolor="#2c3e50")
        
        sizes = [odd_count, even_count]
        labels = [f"ODD\n({odd_count}/6)", f"EVEN\n({even_count}/6)"]
        colors = ["#3498db", "#2ecc71"]
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                           explode=explode, startangle=90, textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        canvas = FigureCanvasTkAgg(fig, master=self.parity_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def draw_size_graph(self):
        """Draw small vs big pie chart"""
        small_count = sum(1 for d in self.draws if 1 <= d <= 18)
        big_count = sum(1 for d in self.draws if 19 <= d <= 36)
        
        # Clear previous canvas
        for widget in self.size_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(4, 2.5), dpi=80, facecolor="#2c3e50")
        ax = fig.add_subplot(111, facecolor="#2c3e50")
        
        sizes = [small_count, big_count]
        labels = [f"SMALL\n({small_count}/6)", f"BIG\n({big_count}/6)"]
        colors = ["#f39c12", "#9b59b6"]
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                           explode=explode, startangle=90, textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        canvas = FigureCanvasTkAgg(fig, master=self.size_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def analyze_color(self):
        if not self.draws:
            return "No draws yet", ""
        
        red_count = sum(1 for d in self.draws if d in self.red_numbers)
        black_count = sum(1 for d in self.draws if d in self.black_numbers)
        zero_count = sum(1 for d in self.draws if d == 0)
        
        drawn_red = any(d in self.red_numbers for d in self.draws)
        drawn_black = any(d in self.black_numbers for d in self.draws)
        
        # Check recent pattern (last 4 draws)
        recent_draws = self.draws[-4:] if len(self.draws) >= 4 else self.draws
        recent_red = sum(1 for d in recent_draws if d in self.red_numbers)
        recent_black = sum(1 for d in recent_draws if d in self.black_numbers)
        
        analysis = f"Red drawn: {red_count}/6 | Black drawn: {black_count}/6 | Zero: {zero_count}/6\n"
        analysis += f"Recent (last 4): Red {recent_red} | Black {recent_black}\n"
        
        if not drawn_red:
            analysis += "✅ RED hasn't been drawn - RECOMMENDED\n"
            rec = "RED"
        elif not drawn_black:
            analysis += "✅ BLACK hasn't been drawn - RECOMMENDED\n"
            rec = "BLACK"
        elif red_count < black_count:
            analysis += "⚠️  BLACK is more likely (RED underdrawn)\n"
            rec = "RED"
        elif black_count < red_count:
            analysis += "⚠️  RED is more likely (BLACK underdrawn)\n"
            rec = "BLACK"
        else:
            # Counts are equal - check recent pattern
            if recent_red > recent_black:
                analysis += "📈 RED trending (recent pattern)\n"
                rec = "RED"
            elif recent_black > recent_red:
                analysis += "📈 BLACK trending (recent pattern)\n"
                rec = "BLACK"
            else:
                analysis += "⚖️  Both equally likely\n"
                rec = "Either"
        
        return analysis, rec
    
    def analyze_parity(self):
        if not self.draws:
            return "No draws yet", ""
        
        odd_count = sum(1 for d in self.draws if d % 2 == 1)
        even_count = sum(1 for d in self.draws if d % 2 == 0)
        
        drawn_odd = any(d % 2 == 1 for d in self.draws)
        drawn_even = any(d % 2 == 0 for d in self.draws)
        
        # Check recent pattern (last 4 draws)
        recent_draws = self.draws[-4:] if len(self.draws) >= 4 else self.draws
        recent_odd = sum(1 for d in recent_draws if d % 2 == 1)
        recent_even = sum(1 for d in recent_draws if d % 2 == 0)
        
        analysis = f"Odd drawn: {odd_count}/6 | Even drawn: {even_count}/6\n"
        analysis += f"Recent (last 4): Odd {recent_odd} | Even {recent_even}\n"
        
        if not drawn_odd:
            analysis += "✅ ODD hasn't been drawn - RECOMMENDED\n"
            rec = "ODD"
        elif not drawn_even:
            analysis += "✅ EVEN hasn't been drawn - RECOMMENDED\n"
            rec = "EVEN"
        elif odd_count < even_count:
            analysis += "⚠️  EVEN is more likely (ODD underdrawn)\n"
            rec = "ODD"
        elif even_count < odd_count:
            analysis += "⚠️  ODD is more likely (EVEN underdrawn)\n"
            rec = "EVEN"
        else:
            # Counts are equal - check recent pattern
            if recent_odd > recent_even:
                analysis += "📈 ODD trending (recent pattern)\n"
                rec = "ODD"
            elif recent_even > recent_odd:
                analysis += "📈 EVEN trending (recent pattern)\n"
                rec = "EVEN"
            else:
                analysis += "⚖️  Both equally likely\n"
                rec = "Either"
        
        return analysis, rec
    
    def analyze_size(self):
        if not self.draws:
            return "No draws yet", ""
        
        small_count = sum(1 for d in self.draws if 1 <= d <= 18)
        big_count = sum(1 for d in self.draws if 19 <= d <= 36)
        zero_count = sum(1 for d in self.draws if d == 0)
        
        drawn_small = any(1 <= d <= 18 for d in self.draws)
        drawn_big = any(19 <= d <= 36 for d in self.draws)
        
        # Check recent pattern (last 4 draws)
        recent_draws = self.draws[-4:] if len(self.draws) >= 4 else self.draws
        recent_small = sum(1 for d in recent_draws if 1 <= d <= 18)
        recent_big = sum(1 for d in recent_draws if 19 <= d <= 36)
        
        analysis = f"Small (1-18): {small_count}/6 | Big (19-36): {big_count}/6 | Zero: {zero_count}/6\n"
        analysis += f"Recent (last 4): Small {recent_small} | Big {recent_big}\n"
        
        if not drawn_small:
            analysis += "✅ SMALL hasn't been drawn - RECOMMENDED\n"
            rec = "SMALL"
        elif not drawn_big:
            analysis += "✅ BIG hasn't been drawn - RECOMMENDED\n"
            rec = "BIG"
        elif small_count < big_count:
            analysis += "⚠️  BIG is more likely (SMALL underdrawn)\n"
            rec = "SMALL"
        elif big_count < small_count:
            analysis += "⚠️  SMALL is more likely (BIG underdrawn)\n"
            rec = "BIG"
        else:
            # Counts are equal - check recent pattern
            if recent_small > recent_big:
                analysis += "📈 SMALL trending (recent pattern)\n"
                rec = "SMALL"
            elif recent_big > recent_small:
                analysis += "📈 BIG trending (recent pattern)\n"
                rec = "BIG"
            else:
                analysis += "⚖️  Both equally likely\n"
                rec = "Either"
        
        return analysis, rec
    
    def get_priority_score(self, category, value):
        """Calculate priority score - higher means more urgent to bet on"""
        if not value or value == "Either":
            return 999  # Low priority for "Either" case
        
        if category == "color":
            red_count = sum(1 for d in self.draws if d in self.red_numbers)
            black_count = sum(1 for d in self.draws if d in self.black_numbers)
            if value == "RED":
                return red_count
            else:
                return black_count
        elif category == "parity":
            odd_count = sum(1 for d in self.draws if d % 2 == 1)
            even_count = sum(1 for d in self.draws if d % 2 == 0)
            if value == "ODD":
                return odd_count
            else:
                return even_count
        elif category == "size":
            small_count = sum(1 for d in self.draws if 1 <= d <= 18)
            big_count = sum(1 for d in self.draws if 19 <= d <= 36)
            if value == "SMALL":
                return small_count
            else:
                return big_count
        return 0
    
    def update_display(self):
        # Display last 6 draws
        if self.draws:
            history_text = " → ".join(str(d) for d in self.draws)
            self.history_display.config(text=history_text)
        else:
            self.history_display.config(text="No draws yet")
        
        if self.draws and len(self.draws) >= 1:
            try:
                # Draw graphs
                self.draw_color_graph()
                self.draw_parity_graph()
                self.draw_size_graph()
                
                # Analysis
                color_text, color_rec = self.analyze_color()
                parity_text, parity_rec = self.analyze_parity()
                size_text, size_rec = self.analyze_size()
                
                # Filter out empty recommendations
                recommendations = []
                if color_rec:
                    recommendations.append(("Color", color_rec, self.get_priority_score("color", color_rec)))
                if parity_rec:
                    recommendations.append(("Parity", parity_rec, self.get_priority_score("parity", parity_rec)))
                if size_rec:
                    recommendations.append(("Size", size_rec, self.get_priority_score("size", size_rec)))
                
                # Sort by count (ascending - lowest count first)
                recommendations.sort(key=lambda x: x[2])
                
                # Recommendation
                rec_text = f"🎯 Next Pick Recommendations (by priority):\n"
                for i, (category, rec, count) in enumerate(recommendations, 1):
                    rec_text += f"  {i}. {rec}\n"
                self.recommendation.config(text=rec_text)
            except Exception as e:
                print(f"Error updating display: {e}")
                self.recommendation.config(text="Error generating recommendations")
        else:
            # Clear graphs
            for widget in self.color_frame.winfo_children():
                widget.destroy()
            for widget in self.parity_frame.winfo_children():
                widget.destroy()
            for widget in self.size_frame.winfo_children():
                widget.destroy()
            self.recommendation.config(text="Add at least 1 draw to see recommendations")
    
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.draws = data.get('draws', [])[-6:]
            except:
                self.draws = []
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump({'draws': self.draws, 'timestamp': datetime.now().isoformat()}, f)
    
    def clear_history(self):
        if messagebox.askyesno("Confirm", "Clear all history?"):
            self.draws = []
            self.save_history()
            self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = RoulettePredictor(root)
    root.mainloop()

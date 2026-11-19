"""Lactate tracker: log workouts (HR + lactate) and see your lactate threshold curve.

Run with:  python app.py
Sections: 1) database  2) login  3) curve  4) screens
"""
import hashlib
import hmac
import os
import sqlite3
import tkinter as tk
from datetime import date
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lactate.db")
THRESHOLD_MMOL = 4.0  # classic "OBLA" lactate threshold marker


# ---------------------------------------------------------------- 1) DATABASE
def db():
    """Open a connection to the SQLite file."""
    return sqlite3.connect(DB_FILE)


def setup_database():
    """Create the two tables if they don't exist yet."""
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            reps INTEGER NOT NULL,
            distance_m REAL NOT NULL,
            grade_pct REAL NOT NULL,
            rpe REAL NOT NULL,
            sleep_hours REAL NOT NULL)""")
        # One workout has 1+ tests (e.g. a 6x mile workout tested on reps 2, 4 and 6)
        conn.execute("""CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY,
            workout_id INTEGER NOT NULL,
            rep_number INTEGER NOT NULL,
            distance_m REAL NOT NULL,
            pace_sec_per_mile REAL NOT NULL,
            hr REAL NOT NULL,
            lactate REAL NOT NULL)""")


# ------------------------------------------------------------------- 2) LOGIN
def hash_password(password, salt):
    """Salted SHA256, repeated 200,000 times so guessing passwords is slow."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def register(username, password):
    """Create a user. Returns True on success, False if the name is taken."""
    salt = os.urandom(16)  # random salt, different for every user
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, hash_password(password, salt)),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def login(username, password):
    """Returns the user's id if the password is right, otherwise None."""
    with db() as conn:
        row = conn.execute(
            "SELECT id, salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return None
    user_id, salt, stored_hash = row
    if hmac.compare_digest(hash_password(password, salt), stored_hash):
        return user_id
    return None


# ------------------------------------------------------------------- 3) CURVE
def load_tests(user_id):
    """Every test (one row each) with its workout's info, as a pandas table, oldest first."""
    with db() as conn:
        return pd.read_sql_query(
            """SELECT w.id AS workout_id, w.date, t.rep_number, t.distance_m, t.pace_sec_per_mile, t.hr, t.lactate,
                      w.reps, w.grade_pct, w.rpe, w.sleep_hours
               FROM tests t JOIN workouts w ON t.workout_id = w.id
               WHERE w.user_id = ?
               ORDER BY w.date, w.id, t.rep_number""",
            conn,
            params=(user_id,),
        )


def parse_pace(text):
    """Turn '6:30' (min:sec per mile) into seconds. Raises ValueError if it's not that format."""
    minutes, seconds = text.split(":")
    minutes, seconds = int(minutes), int(seconds)
    if not (0 < minutes < 30 and 0 <= seconds < 60):
        raise ValueError("bad pace")
    return minutes * 60 + seconds


def format_pace(seconds):
    """Turn 390 back into '6:30'."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def lactate_model(hr, a, b, c):
    """Lactate stays low then rises sharply: a + b * e^(c * hr)."""
    return a + b * np.exp(c * hr)


def fit_curve(df):
    """Fit the model to the data. Returns (a, b, c) or None if it can't."""
    if len(df) < 4 or df["hr"].nunique() < 3:
        return None
    # Scale HR to ~0-1 so the fit is numerically stable, then scale back.
    hr_min, hr_range = df["hr"].min(), max(df["hr"].max() - df["hr"].min(), 1)
    x = (df["hr"] - hr_min) / hr_range
    try:
        (a, b, c), _ = curve_fit(
            lactate_model, x, df["lactate"],
            p0=[df["lactate"].min(), 0.1, 3.0],
            bounds=([0, 0, 0], [10, 50, 20]),
        )
    except RuntimeError:  # fit didn't converge
        return None
    # Convert the scaled constants back to real HR units.
    return a, b * np.exp(-c * hr_min / hr_range), c / hr_range


def threshold_hr(params):
    """HR where the fitted curve reaches 4 mmol (solve the equation for hr)."""
    a, b, c = params
    if THRESHOLD_MMOL <= a:
        return None
    return float(np.log((THRESHOLD_MMOL - a) / b) / c)


# Dropdown choice -> (column to color by, colorbar label). "Date" is worked out in make_figure.
COLOR_OPTIONS = {
    "Date": (None, "Days since first test"),
    "RPE": ("rpe", "RPE"),
    "Pace": ("pace_sec_per_mile", "Pace (sec/mile)"),
    "Sleep": ("sleep_hours", "Sleep (hours)"),
    "None": (None, None),
}


def make_figure(df, color_by="Date"):
    """Build the matplotlib chart. Returns (figure, text describing the result)."""
    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Heart rate (bpm)")
    ax.set_ylabel("Lactate (mmol/L)")
    ax.set_title("Lactate threshold curve")
    ax.grid(alpha=0.3)

    if df.empty:
        return fig, "No tests yet. Add a workout first."

    # Color the points by the chosen column (or by date, or not at all)
    column, color_label = COLOR_OPTIONS[color_by]
    if color_by == "Date":
        dates = pd.to_datetime(df["date"])
        values = (dates - dates.min()).dt.days
    elif column:
        values = df[column]
    if color_label:
        points = ax.scatter(df["hr"], df["lactate"], c=values, cmap="viridis", label="Tests", zorder=3)
        fig.colorbar(points, ax=ax, label=color_label)
    else:
        ax.scatter(df["hr"], df["lactate"], label="Tests", zorder=3)
    params = fit_curve(df)
    if params is None:
        return fig, f"{len(df)} test(s). Need 4+ at different HRs to draw a curve."

    hrs = np.linspace(df["hr"].min() - 5, df["hr"].max() + 5, 200)
    ax.plot(hrs, lactate_model(hrs, *params), label="Fitted curve")
    ax.axhline(THRESHOLD_MMOL, color="gray", linestyle=":")

    thr = threshold_hr(params)
    if thr is None:
        return fig, "Curve fitted, but no threshold found yet."
    ax.axvline(thr, color="red", linestyle="--", label=f"Threshold ~{thr:.0f} bpm")
    ax.legend()
    return fig, f"Lactate threshold ({THRESHOLD_MMOL:.0f} mmol/L): ~{thr:.0f} bpm  ({len(df)} tests)"


# ---------------------------------------------------------------- 4) SCREENS
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lactate Tracker")
        self.geometry("800x720")
        self.user_id = None
        self.show_login()

    def clear(self):
        """Remove everything on screen before drawing a new screen."""
        for widget in self.winfo_children():
            widget.destroy()

    # ---- login screen
    def show_login(self):
        self.clear()
        self.user_id = None
        frame = ttk.Frame(self, padding=40)
        frame.pack()
        ttk.Label(frame, text="Lactate Tracker", font=("", 20)).grid(columnspan=2, pady=15)
        ttk.Label(frame, text="Username").grid(row=1, column=0, sticky="e")
        ttk.Label(frame, text="Password").grid(row=2, column=0, sticky="e")
        username = ttk.Entry(frame)
        password = ttk.Entry(frame, show="*")
        username.grid(row=1, column=1, pady=3)
        password.grid(row=2, column=1, pady=3)

        def do_login():
            user_id = login(username.get().strip(), password.get())
            if user_id is None:
                messagebox.showerror("Login", "Invalid username or password.")
                return
            self.user_id = user_id
            self.show_menu()

        def do_register():
            name, pw = username.get().strip(), password.get()
            if len(name) < 3 or len(pw) < 8:
                messagebox.showerror("Register", "Username 3+ chars, password 8+ chars.")
            elif register(name, pw):
                messagebox.showinfo("Register", "Account created. You can log in now.")
            else:
                messagebox.showerror("Register", "That username is taken.")

        ttk.Button(frame, text="Log in", command=do_login).grid(row=3, column=0, pady=15)
        ttk.Button(frame, text="Register", command=do_register).grid(row=3, column=1)

    # ---- main menu
    def show_menu(self):
        self.clear()
        frame = ttk.Frame(self, padding=40)
        frame.pack()
        ttk.Label(frame, text="Menu", font=("", 20)).pack(pady=15)
        ttk.Button(frame, text="Visualize Curve", command=self.show_curve).pack(fill="x", pady=5)
        ttk.Button(frame, text="Add Workout", command=self.show_add).pack(fill="x", pady=5)
        ttk.Button(frame, text="View Past Workouts", command=self.show_past).pack(fill="x", pady=5)
        ttk.Button(frame, text="Log out", command=self.show_login).pack(fill="x", pady=20)

    def back_button(self):
        ttk.Button(self, text="< Back to menu", command=self.show_menu).pack(anchor="w", padx=10, pady=10)

    # ---- visualize curve
    def show_curve(self, color_by="Date"):
        self.clear()
        self.back_button()
        # Dropdown: picking a new option redraws this screen with that coloring
        picker = ttk.Combobox(self, values=list(COLOR_OPTIONS), state="readonly", width=8)
        picker.set(color_by)
        picker.bind("<<ComboboxSelected>>", lambda event: self.show_curve(picker.get()))
        ttk.Label(self, text="Color points by:").pack()
        picker.pack()
        fig, message = make_figure(load_tests(self.user_id), color_by)
        ttk.Label(self, text=message, font=("", 13)).pack()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---- add workout
    def show_add(self):
        self.clear()
        self.back_button()
        frame = ttk.Frame(self, padding=10)
        frame.pack()

        # --- workout info (once per workout): (label, default value)
        fields = [
            ("Date (YYYY-MM-DD)", date.today().isoformat()),
            ("Number of reps", ""),
            ("Distance per rep (m)", ""),
            ("Hill grade (%)", "0"),
            ("RPE (1-10)", ""),
            ("Sleep last night (hours)", ""),
        ]
        entries = []
        for row, (label, default) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="e", pady=2)
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=2, padx=8)
            entries.append(entry)

        # --- tests (1 or more per workout): one row of boxes per test
        ttk.Label(frame, text="Tests (lactate taken after a rep)", font=("", 13)).grid(
            row=len(fields), columnspan=2, pady=(15, 2))
        tests_frame = ttk.Frame(frame)
        tests_frame.grid(row=len(fields) + 1, columnspan=2)
        for col, name in enumerate(["Rep #", "Distance (m)", "Pace (m:ss/mi)", "HR (bpm)", "Lactate"]):
            ttk.Label(tests_frame, text=name).grid(row=0, column=col)
        test_rows = []  # each item is a list of 5 entry boxes

        def add_test_row():
            row = len(test_rows) + 1
            boxes = []
            for col in range(5):
                box = ttk.Entry(tests_frame, width=13)
                box.grid(row=row, column=col, padx=2, pady=2)
                boxes.append(box)
            boxes[1].insert(0, entries[2].get())  # prefill distance from the workout's rep distance
            test_rows.append(boxes)

        add_test_row()
        ttk.Button(frame, text="+ Add another test", command=add_test_row).grid(
            row=len(fields) + 2, columnspan=2, pady=5)

        def save():
            values = [e.get().strip() for e in entries]
            try:
                date.fromisoformat(values[0])  # raises ValueError if the date is bad
                reps, dist, grade, rpe, sleep = (float(v) for v in values[1:])
                tests = []
                for boxes in test_rows:
                    rep_no, t_dist, pace, hr, lactate = (b.get().strip() for b in boxes)
                    tests.append((int(rep_no), float(t_dist), parse_pace(pace), float(hr), float(lactate)))
            except ValueError:
                messagebox.showerror("Add workout", "Check your entries: date as YYYY-MM-DD, pace as m:ss, everything else a number.")
                return
            rep_numbers = [t[0] for t in tests]
            if not (reps >= 1 and dist > 0 and 1 <= rpe <= 10 and 0 <= sleep <= 24):
                messagebox.showerror("Add workout", "One of the workout numbers is out of a sensible range.")
                return
            if len(set(rep_numbers)) != len(rep_numbers) or not all(1 <= n <= reps for n in rep_numbers):
                messagebox.showerror("Add workout", "Each test needs a different rep number between 1 and the number of reps.")
                return
            if not all(t[1] > 0 and 40 <= t[3] <= 230 and 0 < t[4] <= 30 for t in tests):
                messagebox.showerror("Add workout", "One of the test numbers is out of a sensible range.")
                return
            with db() as conn:
                cursor = conn.execute(
                    """INSERT INTO workouts (user_id, date, reps, distance_m, grade_pct, rpe, sleep_hours)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.user_id, values[0], int(reps), dist, grade, rpe, sleep),
                )
                workout_id = cursor.lastrowid
                for rep_no, t_dist, pace, hr, lactate in tests:
                    conn.execute(
                        """INSERT INTO tests (workout_id, rep_number, distance_m, pace_sec_per_mile, hr, lactate)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (workout_id, rep_no, t_dist, pace, hr, lactate),
                    )
            messagebox.showinfo("Add workout", f"Saved workout with {len(tests)} test(s)!")
            self.show_menu()

        ttk.Button(frame, text="Save workout", command=save).grid(row=len(fields) + 3, columnspan=2, pady=10)

    # ---- past workouts (one row per test)
    def show_past(self):
        self.clear()
        self.back_button()
        columns = ["date", "rep_number", "distance_m", "pace", "hr", "lactate",
                   "reps", "grade_pct", "rpe", "sleep_hours"]
        table = ttk.Treeview(self, columns=columns, show="headings")
        for col in columns:
            table.heading(col, text=col)
            table.column(col, width=70, anchor="center")
        df = load_tests(self.user_id)
        df["pace"] = df["pace_sec_per_mile"].apply(format_pace)
        workout_of_row = {}  # table row -> the workout it belongs to
        for _, row in df.iloc[::-1].iterrows():  # newest first
            item = table.insert("", "end", values=[row[c] for c in columns])
            workout_of_row[item] = int(row["workout_id"])
        table.pack(fill="both", expand=True, padx=10, pady=10)

        def delete_selected():
            selected = table.selection()
            if not selected:
                return
            workout_id = workout_of_row[selected[0]]
            n_tests = list(workout_of_row.values()).count(workout_id)
            if not messagebox.askyesno(
                "Delete", f"Delete this whole workout and all {n_tests} of its test(s)?"
            ):
                return
            with db() as conn:
                # only delete if the workout belongs to the logged-in user
                conn.execute(
                    "DELETE FROM tests WHERE workout_id IN (SELECT id FROM workouts WHERE id = ? AND user_id = ?)",
                    (workout_id, self.user_id),
                )
                conn.execute("DELETE FROM workouts WHERE id = ? AND user_id = ?", (workout_id, self.user_id))
            self.show_past()  # refresh the table

        ttk.Button(self, text="Delete selected workout", command=delete_selected).pack(pady=(0, 10))


if __name__ == "__main__":
    setup_database()
    App().mainloop()

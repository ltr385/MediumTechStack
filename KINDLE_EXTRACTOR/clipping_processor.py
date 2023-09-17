import os
import re
import pandas as pd
from typing import Literal
from datetime import datetime

from aux import deduplicate_dataframe, clean_dataframe


class ClippingProcessor:
    def __init__(self, dropping_type: Literal['clean', 'similarity'] = 'clean'):
        self.deleted_logs = None
        self.similar_logs = None
        self.translate_dict = {
            'lunes': 'Monday', 'martes': 'Tuesday', 'miércoles': 'Wednesday',
            'jueves': 'Thursday', 'viernes': 'Friday', 'sábado': 'Saturday',
            'domingo': 'Sunday', 'enero': 'January', 'febrero': 'February',
            'marzo': 'March', 'abril': 'April', 'mayo': 'May', 'junio': 'June',
            'julio': 'July', 'agosto': 'August', 'septiembre': 'September',
            'octubre': 'October', 'noviembre': 'November',
            'diciembre': 'December'
        }
        self.format_string = 'Añadido el %A, %d de %B de %Y %H:%M:%S'
        self.position_pattern = re.compile(r'posición (\d+)-(\d+)')
        self.dropping_type = dropping_type
        assert dropping_type in ['clean',
                                 'similarity'], 'dropping_type must be either "clean" or "similarity"'
        self.df = None

    def _translate_date(self, date_part):
        """Translate Spanish day and month names to English."""
        for es, en in self.translate_dict.items():
            date_part = date_part.replace(es, en)
        return date_part

    def _extract_data(self, clip):
        """Extract data from a single clipping."""
        clip_splitted = clip.replace('\ufeff', '').replace('Ã³', 'ó').split(
            '\n')
        clip_splitted = [x for x in clip_splitted if x != '']

        title = clip_splitted[0]
        pos_and_date = clip_splitted[1]
        pos_and_date_splitted = pos_and_date.split('|')
        i = 2 if len(pos_and_date_splitted) > 2 else 1

        date_part = self._translate_date(pos_and_date_splitted[i].strip())

        try:
            dt_object = datetime.strptime(date_part, self.format_string)
        except:
            dt_object = pd.NaT

        match = self.position_pattern.search(pos_and_date_splitted[i - 1])

        start_pos = int(match.group(1)) if match else None
        end_pos = int(match.group(2)) if match else None

        return {
            'Title': title,
            'Start_Pos': start_pos,
            'End_Pos': end_pos,
            'Date': dt_object,
            'Text': clip_splitted[2]
        }

    def process_clipping(self, clipping_path):
        """Process a single clipping file."""
        with open(clipping_path, 'r', encoding='utf-8') as f:
            content = f.read()

        clippings = content.split("==========")

        data_list = []
        for clip in clippings:
            try:
                data = self._extract_data(clip)
                data_list.append(data)
            except IndexError:
                pass  # Handle the error as you see fit
        return pd.DataFrame(data_list)

    def process_directory(self, directory_path):
        """Process all clippings in a directory."""
        files = [file for file in os.listdir(directory_path) if
                 file.endswith('.txt')]

        df_list = [self.process_clipping(os.path.join(directory_path, file)) for
                   file in files]

        self.df = pd.concat(df_list, ignore_index=True).drop_duplicates()

        # get the author from the title
        self.df['Author'] = self.df['Title'].apply(
            lambda x: re.findall(r'\((.*?)\)', x)[0]
            if len(re.findall(r'\((.*?)\)', x)) > 0
            else None
        )

        # get the first part of the title before the parenthesis
        self.df['Title'] = self.df['Title'].apply(lambda x: x.split('(')[0])

        return self.df

    def drop_duplicates(self):

        if (self.df is not None) or (self.df.empty()):

            if self.dropping_type != 'clean':
                self.df, self.similar_logs, self.deleted_logs = deduplicate_dataframe(
                    self.df)
                return self.df
            else:
                self.df = clean_dataframe(self.df)
                return self.df

        else:
            raise Exception('You have to process directory first.')

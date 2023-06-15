# Park Scraper builds a full dataset by scrapping information from the website

import datetime
import logging
import time

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

PATH_TO_BROWSER = "/usr/lib/chromium-browser/chromedriver"


class ParkScraper:
    """Scrapes parkrun website and builds a dataset for a given event.

    Uses selenium and beautiful soup to browse the parkrun website and read
    historical results for a given parkrun event. Unless otherwise specified,
    it scrapes the previous 52 events, if available.

    Parameters
    ----------
    park_id : string
        Id of the event to scrape. It should be provided exactly as it appears
        on the parkrun url: https://www.parkrun.org.uk/{event_id}/

    avoid_saturday : bool, default=true
        If true, avoids accessing the parkrun website on a Saturday.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).


    Notes
    -----
    We recommend avoiding running this scraper on Saturdays in order to
    remove unnecessary pressure to the parkrun website.

    Examples
    --------
    >>> from park_report.data import ParkScrapper

    >>> rothay_scraper = ParkScraper(park_id="rothaypark")
    >>> rothay_scraper.fetch_results(from_event=1, to_event=5)
    >>> rothay_results = rothay_scraper.get_results()

    >>> rhuntingdon_scraper = ParkScraper(park_id="huntingdon")
    >>> huntingdon_scraper.fetch_results(last_events=12)
    >>> huntingdon_results = rhuntingdon_scraper.get_results()
    """

    def __init__(self, park_id: str, avoid_saturday: bool = True):
        self._avoid_saturday = avoid_saturday
        self._park_id = park_id

        # Validate parkrun event id
        self._browser = self._init_browser()
        (
            self._event_name,
            self._last_event_no,
            self._event_stats,
        ) = self._fetch_event_summary()

        # Things for later
        self._fetched_events = set([])
        self._dataset = None

    @staticmethod
    def _is_today_saturday():
        today = datetime.date.today()
        return today.weekday() == 5  # 5 corresponds to Saturday

    def _init_browser(self):
        if not self._avoid_saturday or self._is_today_saturday():
            raise Exception(
                """Today is parkrun day (previously known as Saturday). 
            Why don't you wait until tomorrow before doing any scraping?
            Parkrun website will thank you for it. #loveparkrun"""
            )

        options = webdriver.ChromeOptions()
        return webdriver.Chrome(executable_path=PATH_TO_BROWSER, chrome_options=options)

    def _fetch_event_summary(self):
        homepage_url = f"https://www.parkrun.org.uk/{self._park_id}/"
        homepage_soup = self._fetch_page(homepage_url)

        return self._extract_homepage_info(homepage_soup)

    def _fetch_page(self, url: str):
        self._browser.get(url)
        site = self._browser.page_source
        return BeautifulSoup(site, "html.parser")

    def _extract_homepage_info(self, soup):
        # Event name
        event_name = soup.find("h1", {"class", "paddetandb"}).get_text().strip()

        # Event stats
        footer_stats = soup.find_all("div", {"class", "aStat"})
        event_stats = dict(
            [s.get_text().strip().replace("\n", "").split(": ") for s in footer_stats]
        )
        last_event_no = event_stats["Events"]

        # Event records
        event_records = soup.find_all("div", {"class", "records"})
        for record in event_records:
            record_type, record_details = record.find_all("span")
            record_type = record_type.get_text().replace(":", "")
            event_stats[record_type] = (
                record_details.get_text().strip().replace("\n", "")
            )

        return event_name, int(last_event_no), event_stats

    def fetch_results(self, from_event: int = None, to_event: int = None, last_events: int = 12):
        # sourcery skip: use-named-expression
        """Fetches result data from aprkrun website.

        Method either fetches a number of most recent events by passsing `last_events`
        or fetches all events between `from_event` to `to_event`. These last have priority
        over `last_events`.

        Default behaviour will fetch the last 12 events for a given parkrun event.

        The method checks that that event number has not already been fetched before
        making a call to the scraper.


        Parameters
        ----------
        last_events : int, default=12
            Number of most recent events to fetch.

        from_event : int
            Fetch all events with event number >= `from_event`.

        to_event : int
            Fetch all events with event number <= `to_event`.

        Returns
        -------
        self : object
            Scraper with fetched results.

        """
        # Figure out which events to fetch
        if from_event and to_event:
            events_to_fetch = list(np.arange(from_event, to_event + 1))
            # TODO call logger
        else:
            events_to_fetch = [self._last_event_no - n for n in range(last_events)]

        # Have I fetched them already?
        events_to_fetch = list(set(events_to_fetch).difference(self._fetched_events))
        events_to_fetch = [
            event_no for event_no in events_to_fetch if event_no <= self._last_event_no
        ]
        if events_to_fetch:
            self._fetch_event_results(events_to_fetch)
        return self

    def _fetch_event_results(self, events_to_fetch):
        for event_no in events_to_fetch:
            # TODO call logger
            self._fetch_single_event_results(event_no)
            self._fetched_events.add(event_no)
            time.sleep(1)

    def _fetch_single_event_results(self, event_no):
        event_page_url = (
            f"https://www.parkrun.org.uk/{self._park_id}/results/{event_no}"
        )
        event_page_soup = self._fetch_page(event_page_url)

        event_results_df = self._extract_results(event_page_soup)
        event_results_df["event_no"] = event_no

        if self._dataset is None:
            self._dataset = event_results_df
        else:
            self._dataset = pd.concat((self._dataset, event_results_df), ignore_index=True)

    def _extract_results(self, event_soup):
        results = event_soup.find_all("tr", {"class": "Results-table-row"})
        results_data = []

        for result in results:
            this_result = {"name": result.get("data-name")}
            this_result["position"] = result.get("data-position")

            if this_result["name"] != "Unknown":
                # TODO: deal with numeric types
                this_result["parkrun_id"] = (
                    result.find(
                        "td", {"class": "Results-table-td Results-table-td--name"}
                    )
                    .find("a")
                    .get("href")
                    .split("/")[-1]
                )
                this_result["time"] = (
                    result.find("td", {"class": "Results-table-td--time"})
                    .find("div")
                    .get_text()
                )
                this_result["achievement"] = result.get("data-achievement")
                this_result["agegrade"] = result.get("data-agegrade")
                this_result["agegroup"] = result.get("data-agegroup")
                this_result["club"] = result.get("data-club")
                this_result["gender"] = result.get("data-gender")
                this_result["runs"] = result.get("data-runs")

            results_data.append(this_result)

        return pd.DataFrame.from_dict(results_data)
